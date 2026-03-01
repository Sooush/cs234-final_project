"""
SRL (Supervised Reinforcement Learning) training using TRL GRPOTrainer.

Based on paper 2510.25992: "Supervised Reinforcement Learning: From Expert Trajectories
to Step-wise Reasoning". Uses step-wise rewards (similarity between model action step
and expert step) instead of outcome-only rewards.

- Data: SRL instances (problem + previous expert steps -> target next step). Load from
  data/srl_instances.jsonl (from data_prep) or create from s1K-1.1 on the fly.
- Reward: Dense step-wise reward via SequenceMatcher between model's extracted action
  step and expert target step (after parsing <think>...</think> format).
- Trainer: TRL GRPOTrainer (Group Relative Policy Optimization).

Run:
  # Prepare data first (optional if JSONL exists):
  python -m src.data_prep --output data/srl_instances.jsonl

  # Train SRL with LoRA:
  python -m src.train_srl --output-dir checkpoints/srl

  # From base model with custom data:
  python -m src.train_srl --output-dir checkpoints/srl \\
    --data-path data/srl_instances.jsonl --max-train-samples 500
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import GRPOTrainer, GRPOConfig

from .model_config import get_base_model
from .prompts import get_srl_chat_messages, SRL_SYSTEM_PROMPT
from .reward import compute_srl_reward, INVALID_REWARD


DEFAULT_DATASET = "simplescaling/s1K-1.1"


def parse_expert_steps(solution: str) -> List[str] | None:
    """Parse expert solution into numbered steps. Returns list of step strings or None."""
    if not solution or not isinstance(solution, str):
        return None
    parts = re.split(r"(?m)^\s*(\d+)\.\s*", solution)
    if len(parts) < 2:
        return None
    steps = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            step_text = parts[i + 1].strip()
            if step_text:
                steps.append(step_text)
    return steps if len(steps) >= 2 else None


def load_srl_instances_from_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load SRL instances from JSONL produced by data_prep."""
    path = Path(path)
    if not path.exists():
        return []
    instances = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
    return instances


def create_srl_instances_from_s1k(
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    max_examples: int | None = None,
) -> List[Dict[str, Any]]:
    """Create SRL instances from s1K-1.1 (same logic as data_prep)."""
    from .prompts import build_srl_user_prompt

    ds = load_dataset(dataset_name, split=split)
    instances = []
    for idx, item in enumerate(ds):
        if max_examples and len(instances) >= max_examples:
            break
        problem = item.get("problem") or item.get("question") or ""
        solution = item.get("solution") or ""
        if not problem or not solution:
            continue
        steps = parse_expert_steps(solution)
        if not steps:
            continue
        for k in range(2, len(steps) + 1):
            previous_steps = steps[: k - 1]
            target_step = steps[k - 1]
            prompt_user = build_srl_user_prompt(problem, previous_steps)
            instances.append({
                "id": f"s1k_{idx}_k{k}",
                "problem": problem,
                "steps": steps,
                "k": k,
                "prompt_user": prompt_user,
                "target_step": target_step,
            })
            if max_examples and len(instances) >= max_examples:
                break
    return instances


def build_srl_dataset(
    instances: List[Dict[str, Any]],
    tokenizer: Any,
) -> tuple[Dataset, Dict[str, str]]:
    """
    Build HF Dataset with "prompt" column (chat-formatted, with generation prompt).
    Also return prompt -> target_step mapping for the reward function.
    Instances from JSONL have "prompt" (user content) and "target_step"; from s1k have
    "problem", "steps", "k", "prompt_user", "target_step".
    """
    prompts_list: List[str] = []
    prompt_to_target: Dict[str, str] = {}

    for inst in instances:
        if "prompt_user" in inst:
            # From create_srl_instances_from_s1k
            problem = inst["problem"]
            previous_steps = inst.get("steps", [])[: inst["k"] - 1]
            messages = get_srl_chat_messages(problem, previous_steps)
        else:
            # From JSONL (data_prep): "prompt" is user content
            messages = [
                {"role": "system", "content": SRL_SYSTEM_PROMPT},
                {"role": "user", "content": inst["prompt"]},
            ]
        target_step = inst["target_step"]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts_list.append(prompt_text)
        prompt_to_target[prompt_text] = target_step

    dataset = Dataset.from_dict({"prompt": prompts_list})
    return dataset, prompt_to_target


def create_srl_reward_func(prompt_to_target: Dict[str, str]):
    """
    Create reward function for GRPOTrainer: step-wise similarity to expert step.
    TRL calls with (prompts=..., completions=..., **kwargs). Invalid parses get 0.0
    so they don't break advantage computation.
    """
    def reward_func(prompts=None, completions=None, **kwargs):
        if prompts is None:
            prompts = []
        if completions is None:
            completions = []
        rewards = []
        for prompt, completion in zip(prompts, completions):
            target_step = prompt_to_target.get(prompt, "")
            r = compute_srl_reward(completion, target_step)
            if r == INVALID_REWARD:
                r = 0.0
            rewards.append(float(r))
        return rewards
    return reward_func


def main() -> None:
    parser = argparse.ArgumentParser(description="SRL training with TRL GRPOTrainer")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to checkpoint to start from (e.g. SFT or base). Default: base model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/srl",
        help="Directory to save SRL checkpoints",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to SRL instances JSONL (from data_prep). If not set, create from --dataset-name.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET,
        help="HuggingFace dataset for creating SRL instances if --data-path not set",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Max SRL instances for training",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=100,
        help="Max instances for eval (if splitting from train)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume from this checkpoint directory (e.g. checkpoints/srl/checkpoint-50)",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest checkpoint in --output-dir (finds checkpoint-* with largest step)",
    )
    args = parser.parse_args()

    base_model = get_base_model()
    init_from = (args.init_from or "").strip() or base_model

    # Resolve resume: GRPOTrainer.train() has no resume_from_checkpoint, so we resume by loading from checkpoint.
    resume_path = None
    if args.resume_from_checkpoint:
        resume_path = str(Path(args.resume_from_checkpoint).resolve())
    elif args.resume_latest:
        output_path = Path(args.output_dir)
        checkpoints = [p for p in output_path.glob("checkpoint-*") if p.is_dir()]
        def step_num(p):
            s = p.name.split("-")[-1]
            return int(s) if s.isdigit() else -1
        if checkpoints:
            resume_path = str(max(checkpoints, key=step_num))
            print("Resume (latest): %s" % resume_path)
        else:
            print("--resume-latest set but no checkpoint-* in %s; starting from init_from." % args.output_dir)
    if resume_path:
        init_from = resume_path

    print(f"Base model: {base_model}")
    print(f"Init from: {init_from}")

    # Load tokenizer from the same source as the model when possible so vocab stays in sync (avoids
    # model producing token ids not in tokenizer.decoder -> None -> TypeError in decode).
    init_path = Path(init_from)
    if not init_path.is_absolute():
        init_path = Path.cwd() / init_path
    tokenizer_loaded_from_init = False
    if init_path.exists():
        # Prefer any dir that looks like a tokenizer (config, vocab, or tokenizer.json)
        has_tok = (
            (init_path / "tokenizer_config.json").exists()
            or (init_path / "tokenizer.json").exists()
            or (init_path / "vocab.json").exists()
        )
        if has_tok:
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(init_path), use_fast=False)
                tokenizer_loaded_from_init = True
                print(f"Loaded tokenizer from {init_from} (match model vocab)")
            except Exception as e:
                print(f"Could not load tokenizer from {init_from}: {e}; using base model tokenizer.")
    if not tokenizer_loaded_from_init:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        if init_path.exists():
            print("Using tokenizer from base model (no tokenizer files in init_from).")
        else:
            print("Using tokenizer from base model (init_from is not a local path).")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load or create SRL instances
    if args.data_path and Path(args.data_path).exists():
        instances = load_srl_instances_from_jsonl(args.data_path)
        print(f"Loaded {len(instances)} SRL instances from {args.data_path}")
    else:
        instances = create_srl_instances_from_s1k(
            dataset_name=args.dataset_name,
            split="train",
            max_examples=args.max_train_samples,
        )
        print(f"Created {len(instances)} SRL instances from {args.dataset_name}")

    if not instances:
        raise SystemExit("No SRL instances. Run data_prep or use --dataset-name with step-formatted solutions.")

    if args.max_train_samples is not None:
        instances = instances[: args.max_train_samples]

    # Train/eval split
    eval_size = min(args.max_eval_samples or 0, max(0, len(instances) - 10))
    train_size = len(instances) - eval_size
    train_instances = instances[:train_size]
    eval_instances = instances[train_size:] if eval_size > 0 else []

    train_dataset, prompt_to_target = build_srl_dataset(train_instances, tokenizer)
    if eval_instances:
        eval_dataset, eval_prompt_to_target = build_srl_dataset(eval_instances, tokenizer)
        prompt_to_target.update(eval_prompt_to_target)
    else:
        eval_dataset = None

    reward_func = create_srl_reward_func(prompt_to_target)

    # Load model (base or full checkpoint; if init_from is PEFT adapter, load base then adapter)
    load_from_path = init_from
    is_peft_dir = (Path(load_from_path) / "adapter_config.json").exists() if Path(load_from_path).exists() else False

    if is_peft_dir:
        print(f"Loading base model from {base_model}, then PEFT adapter from {load_from_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(model, load_from_path, is_trainable=True)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                load_from_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            print(f"Loaded model from {load_from_path}")
        except Exception as e:
            print(f"Failed to load {load_from_path}: {e}; using base model")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )

    if not args.no_lora:
        # Apply LoRA (skip if already a PEFT model from adapter load)
        if not is_peft_dir:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            try:
                first_layer = next(iter(model.model.layers)) if hasattr(model, "model") else None
                if first_layer and hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                    target_modules.extend(["gate_proj", "up_proj", "down_proj"])
            except Exception:
                pass
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            print(f"Applying LoRA rank={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
            print(f"Target modules: {target_modules}")
            model = get_peft_model(model, lora_config)
        else:
            print("Using existing PEFT adapter (trainable); LoRA args ignored.")
        model.print_trainable_parameters()

    # Enable gradient checkpointing to reduce memory usage
    print("Disabling use_cache and enabling gradient checkpointing")
    model.use_cache = False
    model.gradient_checkpointing_enable()

    # 3. Model loading config for GRPOConfig (if needed)
    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "low_cpu_mem_usage": True,
    }

    per_device_batch = 1
    gradient_accumulation_steps = max(1, 4 // per_device_batch)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=1e-6,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=2,
        num_generations=4,
        max_completion_length=512,
        temperature=1.0,
        beta=0.0,
        model_init_kwargs=model_kwargs,
        logging_steps=10,
        logging_first_step=True,
        log_level="info",
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=3,
        run_name="srl_grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("SRL GRPO Training (step-wise reward, TRL GRPOTrainer)")
    print("=" * 60)
    print(f"Train instances: {len(train_dataset)}")
    print(f"Eval instances: {len(eval_dataset) if eval_dataset else 0}")
    print("=" * 60 + "\n")

    if args.resume_from_checkpoint:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elif args.resume_latest:
        output_path = Path(args.output_dir)
        checkpoints = [p for p in output_path.glob("checkpoint-*") if p.is_dir()]
        def step_num(p):
            s = p.name.split("-")[-1]
            return int(s) if s.isdigit() else -1
        if not checkpoints:
            print("--resume-latest set but no checkpoint-* found in %s; starting from scratch." % args.output_dir)
            train_result = trainer.train()
        else:
            latest = max(checkpoints, key=step_num)
            print("Resuming from latest checkpoint: %s" % latest)
            train_result = trainer.train(resume_from_checkpoint=str(latest))
    else:
        train_result = trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if not args.no_lora and hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir + "_merged")
        tokenizer.save_pretrained(args.output_dir + "_merged")
        print(f"Merged model saved to {args.output_dir}_merged")
    print(f"SRL checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
