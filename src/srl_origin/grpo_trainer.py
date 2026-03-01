"""
GRPO (Group Relative Policy Optimization) trainer for SRL.
Samples G rollouts per prompt; computes group-normalized advantages; applies clipped policy gradient.
Dynamic sampling: filter prompts where std(rewards) < eps_std.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .prompts import get_srl_chat_messages
from .reward import compute_srl_reward
from .utils import load_jsonl_list, set_seed


def _get_logger():
    """Return a logger that writes to stdout and flushes after each message (for monitoring in real time)."""
    name = "grpo_trainer"
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)

    class FlushStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    handler = FlushStreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    return log


logger = _get_logger()


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Convert chat messages to text for tokenization."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


class GRPOTrainer:
    """
    GRPO trainer with SRL-specific:
    - step-wise reward (SequenceMatcher on action step)
    - group normalized advantages
    - dynamic sampling (filter low-variance prompts)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        ref_model: Optional[PreTrainedModel] = None,
        *,
        batch_size: int = 4,
        group_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        kl_coef: float = 0.0,
        clip_epsilon: float = 0.2,
        eps_std: float = 1e-4,
        lr: float = 1e-5,
        checkpoint_every: int = 100,
        output_dir: str = "checkpoints",
        seed: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.eps_std = eps_std
        self.checkpoint_every = checkpoint_every
        self.output_dir = output_dir
        if seed is not None:
            set_seed(seed)

        # Get trainable parameters (important for LoRA - only train LoRA params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in model! Check if LoRA is properly configured.")
        
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=lr)
        except ImportError:
            self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    def _generate_rollouts(self, instances: list[dict], device: torch.device) -> list[list[tuple[str, float, torch.Tensor, torch.Tensor]]]:
        """
        For each instance, sample G rollouts in a single batched generate call.
        Returns list of lists: per-instance list of (text, reward, gen_ids, logprobs).
        """
        all_results = []
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for inst_idx, inst in enumerate(instances):
            logger.info("Generating rollouts for instance %d/%d (batch)", inst_idx + 1, len(instances))
            messages = get_srl_chat_messages(inst["problem"], inst.get("steps", [])[: inst["k"] - 1])
            prompt_text = apply_chat_template(self.tokenizer, messages)
            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            prompt_len = prompt_ids.shape[1]

            # Repeat prompt G times and generate all rollouts in one call
            batched_prompt_ids = prompt_ids.repeat(self.group_size, 1)
            batched_attention_mask = torch.ones_like(batched_prompt_ids)

            with torch.no_grad():
                out = self.model.generate(
                    batched_prompt_ids,
                    attention_mask=batched_attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=pad_id,
                )
                # out: (G, prompt_len + gen_len_padded)
                gen_ids_batch = out[:, prompt_len:].cpu()  # move to CPU immediately

                # Compute logprobs one sequence at a time to avoid OOM.
                # Use per-sequence attention mask to correctly exclude padding tokens.
                token_logprobs_list = []
                for g in range(self.group_size):
                    seq = out[g: g + 1]  # (1, full_len)
                    attn_mask = (seq != pad_id).long()
                    seq_out = self.model(input_ids=seq, attention_mask=attn_mask)
                    gen_ids_g = gen_ids_batch[g]  # (gen_len,) on CPU
                    gen_len_g = gen_ids_g.shape[0]
                    logits_g = seq_out.logits[0, -gen_len_g - 1: -1, :]  # (gen_len, vocab)
                    lp_g = F.log_softmax(logits_g, dim=-1)
                    tlp_g = lp_g.gather(1, gen_ids_g.to(device).unsqueeze(-1)).squeeze(-1)  # (gen_len,)
                    # Zero out logprobs at padding positions so they don't affect the loss
                    pad_mask = (gen_ids_g.to(device) == pad_id)
                    tlp_g = tlp_g.masked_fill(pad_mask, 0.0)
                    token_logprobs_list.append(tlp_g.cpu())
                    del seq_out, logits_g, lp_g, tlp_g

            rollouts = []
            print(f"question: {inst['problem']}")
            for g in range(self.group_size):
                gen_ids = gen_ids_batch[g: g + 1]  # (1, gen_len)
                gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                reward = compute_srl_reward(gen_text, inst["target_step"])
                rollouts.append((gen_text, reward, gen_ids, token_logprobs_list[g]))

            all_results.append(rollouts)

        return all_results

    def _compute_advantages(self, rewards_list: list[list[float]]) -> list[list[float]]:
        """Group normalized advantages: A_i = (r_i - mean(r)) / (std(r) + eps).

        """
        advantages = []
        for rewards in rewards_list:
            r = torch.tensor(rewards, dtype=torch.float32)
            eps = 1e-8
            std_r = r.std(correction=0).item()
            if std_r != std_r:  # nan guard
                std_r = 0.0
            if std_r < self.eps_std:
                advantages.append([0.0] * len(rewards))
            else:
                a = ((r - r.mean()) / (std_r + eps)).tolist()
                advantages.append(a)
        return advantages

    def _grpo_loss_and_backward(
        self,
        instances: list[dict],
        rollouts: list[list[tuple[str, float, torch.Tensor, torch.Tensor]]],
        advantages: list[list[float]],
        device: torch.device,
    ) -> tuple[float, int]:
        """
        GRPO clipped objective. For each rollout, loss = -advantage * min(ratio, clip(ratio)).
        ratio = exp(logprob_new - logprob_old). We use current logprobs as new; stored as old at sampling.
        For simplicity: we recompute logprobs in forward pass and treat rollout rewards as targets.
        Standard GRPO: L = -E[A * min(r_t, clip(r_t))] where r_t = pi(a|s)/pi_old(a|s).
        """
        total_loss_scalar = 0.0
        count = 0

        for inst, rollout_list, adv_list in zip(instances, rollouts, advantages):
            messages = get_srl_chat_messages(inst["problem"], inst.get("steps", [])[: inst["k"] - 1])
            prompt_text = apply_chat_template(self.tokenizer, messages)
            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

            for (gen_text, reward, gen_ids, old_logprobs), adv in zip(rollout_list, adv_list):
                if abs(adv) < 1e-9:
                    continue

                gen_ids_dev = gen_ids.to(device)
                if gen_ids_dev.dim() == 1:
                    gen_ids_dev = gen_ids_dev.unsqueeze(0)
                full_ids = torch.cat([prompt_ids, gen_ids_dev], dim=1)
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                attn_mask = (full_ids != pad_id).long()
                
                # Ensure model is in training mode for gradient computation
                self.model.train()
                outputs = self.model(full_ids, attention_mask=attn_mask)
                gen_len = gen_ids_dev.shape[-1]
                logits = outputs.logits[:, -gen_len - 1: -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                new_logprobs = log_probs.gather(2, gen_ids_dev.unsqueeze(-1)).squeeze(-1).squeeze(0)
                # Zero out padding positions so they don't contribute to the loss
                pad_mask = (gen_ids_dev.squeeze(0) == pad_id)
                new_logprobs = new_logprobs.masked_fill(pad_mask, 0.0)

                old_lp = old_logprobs.to(device)
                if old_lp.dim() == 0:
                    old_lp = old_lp.unsqueeze(0)
                if new_logprobs.dim() == 0:
                    new_logprobs = new_logprobs.unsqueeze(0)
                min_len = min(old_lp.shape[0], new_logprobs.shape[0])
                # Per-token ratios then mean — matches Eq. 1 in the SRL paper
                # Summing log-probs then exp() causes exp overflow for long sequences
                log_ratios = new_logprobs[:min_len] - old_lp[:min_len]  # (T,)
                if not torch.isfinite(log_ratios).all():
                    del outputs, logits, log_probs, new_logprobs
                    continue
                per_token_ratio = torch.exp(log_ratios.clamp(-5, 5))  # (T,)
                per_token_clipped = torch.clamp(per_token_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                loss_term = -adv * torch.min(per_token_ratio, per_token_clipped).mean()

                # Verify that loss_term requires gradients before backward
                if not loss_term.requires_grad:
                    raise RuntimeError(
                        f"Loss term does not require gradients! "
                        f"Model in training mode: {self.model.training}, "
                        f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
                    )

                if torch.isfinite(loss_term):
                    loss_term.backward()
                    total_loss_scalar += loss_term.item()
                    count += 1
                del outputs, logits, log_probs, new_logprobs, per_token_ratio, per_token_clipped, loss_term

        return total_loss_scalar, count

    def _apply_dynamic_filter(
        self,
        instances: list[dict],
        rollouts: list[list[tuple[str, float, torch.Tensor, torch.Tensor]]],
    ) -> tuple[list[dict], list[list[tuple[str, float, torch.Tensor, torch.Tensor]]]]:
        """
        Filter out prompts where std(rewards) < eps_std. Return filtered instances and rollouts.
        """
        filtered_inst = []
        filtered_roll = []
        for inst, rollout_list in zip(instances, rollouts):
            rewards = [r for _, r, _, _ in rollout_list]
            std_r = float(torch.tensor(rewards).std().item()) if len(rewards) > 1 else 0.0
            if std_r != std_r:  # nan
                std_r = 0.0
            if std_r >= self.eps_std:
                filtered_inst.append(inst)
                filtered_roll.append(rollout_list)
        return filtered_inst, filtered_roll

    def _sample_until_batch_full(
        self,
        all_instances: list[dict],
        device: torch.device,
    ) -> tuple[list[dict], list[list], int]:
        """
        Continuously sample individual instances and generate rollouts until
        batch_size instances pass the dynamic filter (std(rewards) >= eps_std).
        Returns (filtered_instances, filtered_rollouts, total_candidates_tried).
        Matches the paper: 'continuously sample and filter until the batch is filled'.
        """
        inst_filtered: list[dict] = []
        roll_filtered: list[list] = []
        total_tried = 0

        while len(inst_filtered) < self.batch_size:
            candidate = random.choice(all_instances)
            rollout_list = self._generate_rollouts([candidate], device)[0]
            total_tried += 1
            rewards = [r for _, r, _, _ in rollout_list]
            std_r = float(torch.tensor(rewards).std(correction=0).item()) if len(rewards) > 1 else 0.0
            if std_r != std_r:
                std_r = 0.0
            if std_r >= self.eps_std:
                inst_filtered.append(candidate)
                roll_filtered.append(rollout_list)
            # Progress log every 5 candidates so user can see sampling is active
            if total_tried % 5 == 0:
                logger.info(
                    "Sampling batch: tried %d candidates, accepted %d/%d",
                    total_tried, len(inst_filtered), self.batch_size,
                )

        return inst_filtered, roll_filtered, total_tried

    def train_step(
        self,
        all_instances: list[dict],
        device: torch.device,
    ) -> dict[str, float]:
        """One training step: sample until batch full, compute advantages, loss, update."""
        if not all_instances:
            return {"loss": 0.0, "reward_mean": 0.0, "reward_std": 0.0, "filter_rate": 0.0}

        logger.info("Train step: sampling until batch full (batch_size=%d, group_size=%d)...", self.batch_size, self.group_size)
        inst_filtered, roll_filtered, total_tried = self._sample_until_batch_full(all_instances, device)
        logger.info("Train step: batch filled after %d candidates; computing advantages and loss.", total_tried)
        filter_rate = 1.0 - self.batch_size / max(1, total_tried)

        rewards_flat = []
        for rl in roll_filtered:
            rewards_flat.extend([r for _, r, _, _ in rl])
        reward_mean = sum(rewards_flat) / len(rewards_flat) if rewards_flat else 0.0
        reward_std = (sum((x - reward_mean) ** 2 for x in rewards_flat) / len(rewards_flat)) ** 0.5 if rewards_flat else 0.0
        invalid_rate = sum(1 for r in rewards_flat if r < 0) / len(rewards_flat) if rewards_flat else 0.0

        advantages = self._compute_advantages([[r for _, r, _, _ in rl] for rl in roll_filtered])

        self.model.train()
        self.optimizer.zero_grad()
        total_loss_scalar, count = self._grpo_loss_and_backward(inst_filtered, roll_filtered, advantages, device)
        if count > 0:
            # Normalize gradients by count (equivalent to mean loss backward)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.div_(count)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # Skip optimizer step if gradients are nan/inf to prevent weight corruption
        grad_finite = all(
            torch.isfinite(p.grad).all()
            for p in self.model.parameters()
            if p.grad is not None
        )
        if grad_finite:
            self.optimizer.step()
        loss_scalar = total_loss_scalar / count if count > 0 else 0.0

        logger.info(
            "Train step done: loss=%.6f reward_mean=%.4f reward_std=%.4f invalid=%.2f%% filter=%.2f%% grad_norm=%.4f",
            loss_scalar, reward_mean, reward_std, 100 * invalid_rate, 100 * filter_rate, grad_norm.item(),
        )
        return {
            "loss": loss_scalar,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "filter_rate": filter_rate,
            "invalid_rate": invalid_rate,
            "grad_norm": grad_norm.item(),
        }

    def validate(self, val_instances: list[dict], device: torch.device, num_samples: int = 64) -> dict[str, float]:
        """
        Evaluate on a fixed val set by generating rollouts and computing mean reward.
        No gradient updates. Uses a fixed random seed for reproducibility across calls.
        """
        logger.info("Validation: sampling %d instances", min(num_samples, len(val_instances)))
        self.model.eval()
        rng_state = random.getstate()
        random.seed(0)
        sample = random.sample(val_instances, min(num_samples, len(val_instances)))
        random.setstate(rng_state)

        all_rewards: list[float] = []
        with torch.no_grad():
            for inst in sample:
                rollouts = self._generate_rollouts([inst], device)[0]
                all_rewards.extend(reward for _, reward, _, _ in rollouts)

        self.model.train()
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        invalid_rate = sum(1 for r in all_rewards if r < 0) / len(all_rewards) if all_rewards else 0.0
        valid_rewards = [r for r in all_rewards if r >= 0]
        valid_mean = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
        return {"val_reward": mean_reward, "val_valid_reward": valid_mean, "val_invalid": invalid_rate}

    def train(
        self,
        data_path: str,
        num_steps: int,
        device: torch.device,
        start_step: int = 0,
        val_data_path: str | None = None,
        eval_every: int = 50,
    ) -> None:
        """Full training loop with periodic checkpointing and optional validation."""
        instances = load_jsonl_list(data_path)
        if not instances:
            raise ValueError(f"No instances in {data_path}")

        logger.info(
            "GRPO training starting: data_path=%s instances=%d num_steps=%d start_step=%d device=%s output_dir=%s",
            data_path, len(instances), num_steps, start_step, device, self.output_dir,
        )

        val_instances = load_jsonl_list(val_data_path) if val_data_path else []
        if val_instances:
            logger.info("Validation set: %d instances, evaluating every %d steps", len(val_instances), eval_every)
            tqdm.write(f"Validation set: {len(val_instances)} instances, evaluating every {eval_every} steps")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        best_val_reward = -float("inf")
        best_ckpt_path: str | None = None

        for i in tqdm(range(num_steps - start_step), desc="GRPO"):
            step = start_step + i
            logger.info("Starting step %d / %d", step, num_steps)
            metrics = self.train_step(instances, device)
            if step % 10 == 0:
                tqdm.write(
                    f"Step {step}: loss={metrics['loss']:.6f} reward_mean={metrics['reward_mean']:.4f} "
                    f"reward_std={metrics['reward_std']:.4f} invalid={metrics['invalid_rate']:.2%} "
                    f"filter={metrics['filter_rate']:.2%} grad_norm={metrics.get('grad_norm', 0):.4f}"
                )

            if (step + 1) % self.checkpoint_every == 0:
                ckpt_path = f"{self.output_dir}/step_{step+1}"
                logger.info("Saving checkpoint to %s", ckpt_path)
                self.model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                Path(ckpt_path).joinpath("trainer_step.txt").write_text(str(step + 1))
                tqdm.write(f"Checkpoint saved: {ckpt_path}")

            if val_instances and (step + 1) % eval_every == 0:
                logger.info("Running validation at step %d", step + 1)
                val_metrics = self.validate(val_instances, device)
                logger.info(
                    "Val step %d: reward=%.4f valid_reward=%.4f invalid=%.2f%%",
                    step + 1, val_metrics["val_reward"], val_metrics["val_valid_reward"], 100 * val_metrics["val_invalid"],
                )
                tqdm.write(
                    f"  [Val step {step+1}] reward={val_metrics['val_reward']:.4f} "
                    f"valid_reward={val_metrics['val_valid_reward']:.4f} "
                    f"invalid={val_metrics['val_invalid']:.2%}"
                )
                if val_metrics["val_valid_reward"] > best_val_reward:
                    best_val_reward = val_metrics["val_valid_reward"]
                    best_ckpt_path = f"{self.output_dir}/best"
                    logger.info("New best validation reward=%.4f, saving to %s", best_val_reward, best_ckpt_path)
                    self.model.save_pretrained(best_ckpt_path)
                    self.tokenizer.save_pretrained(best_ckpt_path)
                    Path(best_ckpt_path).joinpath("trainer_step.txt").write_text(str(step + 1))
                    tqdm.write(f"  [Val] New best (val_valid_reward={best_val_reward:.4f}) -> {best_ckpt_path}")

        if best_ckpt_path:
            logger.info("Training finished. Best checkpoint: %s (val_valid_reward=%.4f)", best_ckpt_path, best_val_reward)
            tqdm.write(f"Best checkpoint: {best_ckpt_path} (val_valid_reward={best_val_reward:.4f})")
