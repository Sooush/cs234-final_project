"""
RLVR (outcome-based GRPO) training script with LoRA.

Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

- batch size ≈ 128 (prompts)
- learning_rate = 5e-7
- rollout temperature = 1.0
- rollout number (num_generations) = 8
- KL loss coeff (beta) = 0.0
- dtype = bf16 (if available)
- LoRA rank = 16 (default)
- LoRA alpha = 32 (default)

This script can be run:
1. After SRL training, using the SRL checkpoint as initialization:
   python -m src.train_rlvr_grpo \\
     --init-from checkpoints/srl/step_500 \\
     --output-dir checkpoints/srl_rlvr

2. From the base model (if --init-from is not provided):
   python -m src.train_rlvr_grpo \\
     --output-dir checkpoints/srl_rlvr

After training, point `configs/models_config.json["models"]["srl_rlvr"]["model_path"]`
to the RLVR checkpoint directory (e.g. `checkpoints/srl_rlvr` or `checkpoints/srl_rlvr_merged`) 
so that the existing evaluation pipeline can pick it up.
"""

from __future__ import annotations

import os

# Reduce CUDA memory fragmentation (set before importing torch)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig
import torch

from .model_config import get_base_model


DEFAULT_DATASET = "simplescaling/s1K-1.1"


@dataclass
class RLVRConfig:
    init_from: str
    output_dir: str
    dataset_name: str
    max_train_samples: int | None
    max_eval_samples: int | None

    # Paper hyperparameters (Table 6)
    learning_rate: float = 5e-7
    batch_size: int = 4
    num_generations: int = 4
    max_completion_length: int = 8192  # max tokens to generate per completion
    num_train_epochs: int = 1  # paper uses steps; epochs is a practical proxy
    beta: float = 0.0  # KL coeff
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    temperature: float = 1.0
    eval_every: int = 50
    checkpoint_every: int = 5
    dataloader_num_workers: int = 0


def build_prompt(example: Dict[str, Any]) -> str:
    """
    Build the RLVR prompt from a s1K-style example (fields: "problem", "solution").
    Follows the format from paper 2510.25992 (Supervised Reinforcement Learning):
    the model should first produce an internal reasoning monologue in <think>...</think>,
    then provide the step-by-step solution and final answer after </think>.
    Instructions stress token efficiency, a single answer, and no repetition or off-topic content.
    """
    problem = example.get("problem") or example.get("question") or ""
    open_think, close_think = "<think>", "</think>"
    return (
        "You are a helpful math assistant. Solve ONLY the following problem. Be token-efficient: "
        "no repetition, no extra questions, no filler.\n\n"
        "Rules:\n"
        "1. Use exactly one <think>...</think> block with concise reasoning, then brief solution steps.\n"
        "2. End with exactly one line: \"The answer is [number]\" (replace [number] with your final answer).\n"
        "3. Stop immediately after that line. Do not repeat the answer, do not answer other questions, "
        "and do not add more text to fill space.\n\n"
        "Format: <think> key steps only </think> solution steps, then \"The answer is [number]\".\n\n"
        f"Problem:\n{problem}\n\n"
        "Answer:"
    )


def extract_final_answer(text: str) -> str:
    """
    Heuristic extractor for final scalar answer from model output.
    For a real project, you may want to reuse the AIME-style extractor in utils.py.
    """
    # Very simple heuristic: take the last number in the string.
    import re

    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


def _answers_match(gt: str, pred: str) -> bool:
    """Compare extracted answers; treat numeric equality so 5 == 5.0."""
    if not gt or not pred:
        return False
    if gt == pred:
        return True
    try:
        return float(gt) == float(pred)
    except ValueError:
        return False

FORMAT_REWARD_VALUE = 0.2
ACCURACY_REWARD_VALUE = 0.8


def _has_think_format(text: str) -> bool:
    """
    True if the completion has <think>...</think> format (opening and closing think tags
    with closing tag after the opening, and some content in or after the block).
    RLVR-specific check; does not use SRL parsing.
    """
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    open_tag, close_tag = "<think>", "</think>"
    if open_tag not in t or close_tag not in t:
        return False
    start = t.find(open_tag)
    end = t.find(close_tag, start)
    if end == -1:
        return False
    think_content = t[start + len(open_tag) : end].strip()
    after_think = t[end + len(close_tag) :].strip()
    return bool(think_content or after_think)


def create_accuracy_reward_func(dataset, prompt_builder=None):
    """
    Create an accuracy reward function that has access to the dataset for ground truth.
    
    The GRPOTrainer in TRL v0.28.0 calls reward functions with:
      - prompts: list of prompt strings (keyword argument)
      - completions: list of model-generated completion strings (keyword argument)
      - Additional kwargs may contain metadata
    """
    # Default prompt builder falls back to build_prompt if none is provided.
    if prompt_builder is None:
        prompt_builder = build_prompt

    # Create a mapping from prompt to ground truth completion.
    # Use stripped prompt as key so lookup works if the trainer passes slightly different whitespace.
    prompt_to_gt = {}
    for example in dataset:
        prompt = prompt_builder(example)
        key = prompt.strip() if prompt else ""
        gt_completion = example.get("solution", "")
        prompt_to_gt[key] = gt_completion
    
    def accuracy_reward_func(prompts=None, completions=None, **kwargs):
        """
        Simple RLVR reward: 0.8 if final answer matches ground truth, else 0.0.
        Note: If all rewards are 0, GRPO advantages become 0 (zero reward std per group),
        so training loss will be 0 and no learning happens. Ensure prompt lookup and
        answer comparison are correct so some rewards can be non-zero.
        """
        if prompts is None:
            prompts = []
        if completions is None:
            completions = []

        rewards: list[float] = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            # Look up with stripped key to match how we built the map
            key = (prompt.strip() if prompt else "")
            ground_truth = prompt_to_gt.get(key, "")

            gt = extract_final_answer(ground_truth)
            pred = extract_final_answer(completion)
            rewards.append(ACCURACY_REWARD_VALUE if _answers_match(gt, pred) else 0.0)

        # Optional: log first few reward samples to verify lookup and comparison (set RLVR_DEBUG_REWARD=1)
        if os.environ.get("RLVR_DEBUG_REWARD", "").strip() == "1":
            for i, (r, p, c) in enumerate(zip(rewards, prompts or [], completions or [])):
                if i >= 3:
                    break
                key = (p.strip() if p else "")
                gt_raw = prompt_to_gt.get(key, "")
                gt = extract_final_answer(gt_raw)
                pred = extract_final_answer(c)
                print(f"[RLVR reward] sample {i}: gt={gt!r} pred={pred!r} reward={r} (key_in_map={key[:50]!r}...)")
            os.environ["RLVR_DEBUG_REWARD"] = "0"  # log only once

        try:
            with open("/tmp/rlvr_reward.log", "a") as f:
                f.write(f"Accuracy Reward: {rewards[-1]}\n")
        except OSError:
            pass
        return rewards
    
    return accuracy_reward_func





def create_format_reward_func():
    """
    Create a format reward function for GRPO.
    Returns FORMAT_REWARD_VALUE (0.2) per completion that has correct format.
    Debug: rewards are appended to /tmp/rlvr_format_reward.log.
    """

    def format_reward_func(prompts=None, completions=None, **kwargs):
        if completions is None:
            completions = []
        rewards: list[float] = []
        for completion in completions:
            rewards.append(FORMAT_REWARD_VALUE if _has_think_format(completion) else 0.0)
            try:
                with open("/tmp/rlvr_reward.log", "a") as f:
                    f.write(f"Format Reward: {rewards[-1]}\n Completion: {completion}\n")
            except OSError:
                pass
        return rewards

    return format_reward_func


def main() -> None:
    parser = argparse.ArgumentParser(description="RLVR GRPO training (SRL → RLVR)")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to SRL checkpoint to initialize RLVR (e.g. checkpoints/srl/step_500). "
             "If not provided or empty, defaults to base model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save RLVR checkpoints",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET,
        help="HF dataset name (default: simplescaling/s1K-1.1)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional: limit number of training samples (for debugging)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional: limit number of eval samples",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha scaling parameter (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="LoRA dropout rate (default: 0.05)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and use full fine-tuning",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory to resume RLVR training from "
            "(e.g. checkpoints/srl_rlvr/checkpoint-50). "
            "If not set, training starts from scratch in output_dir."
        ),
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=None,
        help="Max tokens to generate per completion (default: 8192)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Number of generations",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="KL coefficient",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Evaluation interval in steps (optional; can be set via config YAML).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Checkpoint save interval in steps (optional; can be set via config YAML).",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=None,
        help="Number of dataloader worker processes (optional; can be set via config YAML).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rlvr_l4.yaml",
        help="Path to YAML config file to override training hyperparameters (e.g. configs/rlvr_l4.yaml).",
    )
    args = parser.parse_args()

    # Load YAML config (if provided) to override defaults
    yaml_cfg: Dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                yaml_cfg = loaded
                print(f"Loaded RLVR config from {cfg_path}")
            else:
                print(f"Config file {cfg_path} did not contain a dict; ignoring.")
        else:
            print(f"Config file {cfg_path} not found; using CLI/default hyperparameters.")

    # Override args from config for any argument not explicitly provided on CLI (i.e., still None).
    # Config values take precedence; hardcoded fallbacks apply when neither CLI nor config provides a value.
    # Use setattr/getattr because argparse.Namespace attributes are dynamic and not statically known.
    if getattr(args, "output_dir", None) is None:
        setattr(args, "output_dir", str(yaml_cfg.get("output_dir", "checkpoints/rlvr_l4_token_8192/")))
    if getattr(args, "max_train_samples", None) is None and "max_train_samples" in yaml_cfg:
        setattr(args, "max_train_samples", yaml_cfg["max_train_samples"])
    if getattr(args, "max_eval_samples", None) is None and "max_eval_samples" in yaml_cfg:
        setattr(args, "max_eval_samples", int(yaml_cfg.get("max_eval_samples", 50)))
    if getattr(args, "max_completion_length", None) is None and "max_new_tokens" in yaml_cfg:
        setattr(args, "max_completion_length", int(yaml_cfg.get("max_new_tokens", 8192)))
    if getattr(args, "lora_r", None) is None and "lora_r" in yaml_cfg:
        setattr(args, "lora_r", int(yaml_cfg.get("lora_r", 32)))
    if getattr(args, "lora_alpha", None) is None and "lora_alpha" in yaml_cfg:
        setattr(args, "lora_alpha", int(yaml_cfg.get("lora_alpha", 64)))
    if getattr(args, "lora_dropout", None) is None and "lora_dropout" in yaml_cfg:
        setattr(args, "lora_dropout", float(yaml_cfg.get("lora_dropout", 0.05)))
    if getattr(args, "learning_rate", None) is None and "lr" in yaml_cfg:
        setattr(args, "learning_rate", float(yaml_cfg.get("lr", 5e-7)))
    if getattr(args, "batch_size", None) is None and "batch_size" in yaml_cfg:
        setattr(args, "batch_size", int(yaml_cfg.get("batch_size", 4)))
    if getattr(args, "num_generations", None) is None and "num_generations" in yaml_cfg:
        setattr(args, "num_generations", int(yaml_cfg.get("num_generations", 4)))
    if getattr(args, "num_train_epochs", None) is None and "num_train_epochs" in yaml_cfg:
        setattr(args, "num_train_epochs", int(yaml_cfg.get("num_train_epochs", 1)))
    if getattr(args, "beta", None) is None and "kl_coef" in yaml_cfg:
        setattr(args, "beta", float(yaml_cfg.get("kl_coef", 0.0)))
    if getattr(args, "temperature", None) is None and "temperature" in yaml_cfg:
        setattr(args, "temperature", float(yaml_cfg.get("temperature", 1.0)))
    if getattr(args, "eval_every", None) is None and "eval_every" in yaml_cfg:
        setattr(args, "eval_every", int(yaml_cfg.get("eval_every", 50)))
    if getattr(args, "checkpoint_every", None) is None and "checkpoint_every" in yaml_cfg:
        setattr(args, "checkpoint_every", int(yaml_cfg.get("checkpoint_every", 5)))
    if getattr(args, "dataloader_num_workers", None) is None and "dataloader_num_workers" in yaml_cfg:
        setattr(args, "dataloader_num_workers", int(yaml_cfg.get("dataloader_num_workers", 0)))

    cfg = RLVRConfig(
        init_from=args.init_from,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        temperature=args.temperature,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    base_model = yaml_cfg.get("model", get_base_model())
    print(f"Base model (for tokenizer): {base_model}")
    
    # Default to base model if --init-from is not provided or is empty
    if not args.init_from or args.init_from.strip() == "":
        init_from = base_model
        print(f"Initializing RLVR from base model: {init_from}")
    else:
        init_from = args.init_from
        print(f"Initializing RLVR from SRL checkpoint: {init_from}")
    
    # Update cfg with the determined init_from path
    cfg.init_from = init_from

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load dataset
    # Note: s1K-1.1 only has "train" split, so we split it manually.
    # Shuffle the full dataset once so both train and eval are random subsets.
    dataset = load_dataset(cfg.dataset_name, split="train").shuffle(seed=42)
    
    # Split train/val from the shuffled dataset (same approach as train_sft.py)
    EVAL_SIZE = 60
    TRAIN_SIZE = len(dataset) - EVAL_SIZE
    
    raw_train = dataset.select(range(TRAIN_SIZE))
    raw_eval = dataset.select(range(TRAIN_SIZE, (TRAIN_SIZE + EVAL_SIZE)))

    if cfg.max_train_samples is not None:
        raw_train = raw_train.select(range(min(cfg.max_train_samples, len(raw_train))))
    if cfg.max_eval_samples is not None:
        raw_eval = raw_eval.select(range(min(cfg.max_eval_samples, len(raw_eval))))

    # Build prompts using the tokenizer's chat template, with a separate system and user message.
    def build_chat_prompt(example: Dict[str, Any]) -> str:
        problem_text = example.get("problem") or example.get("question") or ""
        open_think, close_think = "<think>", "</think>"

        system_content = (
            "You are a helpful assistant for solving mathematical problems.\n"
            "A user will provide a math problem and ask you to solve the task.\n"
            "You should first draft your thinking process (inner monologue). Then, generate the solution."
            "Your response format must follow the template below: <think> Your thoughts or/and draft, like working through an exercise on"
            "scratch paper. Be as casual and as long as you want until you are confident to generate a correct solution. </think>\n"
            "Provide only the single answer and solve the problem.\n"

            "You are a helpful assistant specialized in solving mathematical problems.\n"
            "A user will provide a math problem for you to solve.\n"
            "You must first outline your reasoning process (inner monologue) in detail, then provide the final solution.\n"
            "Response Format Requirement:\n"
            "Your response must strictly follow this template:\n"
            "<think> [Your step-by-step reasoning, calculations, and scratchpad notes. Be as thorough and informal as needed until you are confident in the result.] </think>\n"
            "After the think block, provide only the final answer without further explanation."
        )
        user_content = f"Here is the problem:\n{problem_text}\n\nPlease provide the answer:"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def map_to_prompts(example: Dict[str, Any]) -> Dict[str, Any]:
        # GRPOTrainer expects specific column names in TRL v0.28.0
        # Use standard column names: "prompt" and "completion"
        return {
            "prompt": build_chat_prompt(example),
            "completion": example.get("solution", ""),  # Use "completion" instead of "solution"
        }

    train_dataset = raw_train.map(map_to_prompts)
    eval_dataset = raw_eval.map(map_to_prompts)

    # Create reward function with access to the dataset for ground truth matching.
    # Use the same chat-formatted prompt builder so lookup keys match GRPOTrainer prompts.
    accuracy_reward_func = create_accuracy_reward_func(raw_train, prompt_builder=build_chat_prompt)
    format_reward_func = create_format_reward_func()

    # 2. Load model and apply LoRA (if enabled)
    # Try to load from init_from, fallback to base model if it fails
    print(f"Attempting to load model from: {init_from}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            init_from,
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"Successfully loaded model from: {init_from}")
    except Exception as e:
        print(f"Failed to load model from '{init_from}': {e}")
        print(f"Falling back to base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"Successfully loaded base model: {base_model}")
    
    if not args.no_lora:
        # Configure LoRA for Qwen models
        # Target attention and MLP layers
        print("Applying LoRA")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        # For Qwen models, also target MLP layers if they exist
        try:
            # Check if model has gate_proj, up_proj, down_proj (typical for Qwen)
            first_layer = next(iter(model.model.layers)) if hasattr(model, 'model') else None
            if first_layer and hasattr(first_layer, 'mlp'):
                if hasattr(first_layer.mlp, 'gate_proj'):
                    target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        except Exception as e:
            print(f"Error in target modules: {e}")
            pass
        
        _lora_r = getattr(args, "lora_r", 16)
        _lora_alpha = getattr(args, "lora_alpha", 32)
        _lora_dropout = getattr(args, "lora_dropout", 0.05)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=_lora_r,
            lora_alpha=_lora_alpha,
            lora_dropout=_lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        print(f"Applying LoRA with rank={_lora_r}, alpha={_lora_alpha}, dropout={_lora_dropout}")
        print(f"Target modules: {target_modules}")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Using full fine-tuning (LoRA disabled)")

    # Enable gradient checkpointing to reduce memory usage
    print("Disabling use_cache and enabling gradient checkpointing")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 3. Model loading config for GRPOConfig (if needed)
    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "low_cpu_mem_usage": True,
    }

    # 4. GRPO training configuration (match Table 6)
    # Paper hyperparameters (Table 6):
    # - batch size ≈ 128 (prompts)
    # - learning_rate = 5e-7
    # - rollout temperature = 1.0
    # - rollout number (num_generations) = 8
    # - KL loss coeff (beta) = 0.0
    # - dtype = bf16 (if available)
    
    per_device_train_batch_size = 1
    gradient_accumulation_steps = cfg.batch_size // per_device_train_batch_size
    print(f"Per device train batch size: {per_device_train_batch_size}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # GRPOConfig in TRL v0.28.0 extends TrainingArguments
    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        logging_steps=10,  # Log every 10 steps
        logging_first_step=True,  # Log the first step
        log_level="info",  # Info level logging
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=cfg.num_generations,  # must be divisible by num_generations
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        beta=cfg.beta,  # KL coeff
        model_init_kwargs=model_kwargs,
        report_to="none",  # No external logging (wandb/tensorboard)
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        # Evaluation and saving
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg.eval_every,
        save_strategy="steps",
        save_steps=cfg.checkpoint_every,
        save_total_limit=3,  # Keep only the last 3 checkpoints
        load_best_model_at_end=False,  # Don't load best model (we'll handle this manually if needed)
        # Logging details
        logging_dir=cfg.output_dir,  # Directory for logs
        run_name="rlvr_grpo_training",  # Name for this training run
        dataloader_num_workers=cfg.dataloader_num_workers,
    )

    # 5. Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,  # Use the LoRA-enabled model
        reward_funcs=[accuracy_reward_func, format_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 6. Start training with logging
    print("\n" + "="*80)
    print("Starting RLVR GRPO Training")
    print("="*80)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset) if eval_dataset else 0}")
    print(f"Batch size: {cfg.batch_size} (per_device: {per_device_train_batch_size}, grad_accum: {gradient_accumulation_steps})")
    print(f"Number of generations per prompt: {cfg.num_generations}")
    print(f"Max completion length (tokens): {cfg.max_completion_length}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Number of epochs: {cfg.num_train_epochs}")
    print(f"Temperature: {cfg.temperature}")
    print(f"Beta (KL coeff): {cfg.beta}")
    print(f"Max train samples: {cfg.max_train_samples}")
    print(f"Max eval samples: {cfg.max_eval_samples}")
    print(f"LoRA r: {cfg.lora_r}")
    print(f"LoRA alpha: {cfg.lora_alpha}")
    print(f"LoRA dropout: {cfg.lora_dropout}")
    print(f"Eval every: {cfg.eval_every} steps")
    print(f"Checkpoint every: {cfg.checkpoint_every} steps")
    print(f"Dataloader number of workers: {cfg.dataloader_num_workers}")
    print("="*80 + "\n")
    
    # Start training - logs will be displayed automatically
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        train_result = trainer.train()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Training loss: {train_result.training_loss:.4f}" if hasattr(train_result, 'training_loss') else "")
    if hasattr(train_result, 'metrics'):
        print(f"Training metrics: {train_result.metrics}")
    print("="*80 + "\n")

    # 7. Save final model (useful path for evaluation pipeline)
    if not args.no_lora:
        # For LoRA, save the adapter weights
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        
        # Also merge and save the full model (for easier evaluation)
        # This creates a model that can be loaded without PEFT
        merged_model = model.merge_and_unload()
        merged_output_dir = cfg.output_dir + "_merged"
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        print(f"Saved LoRA adapter to: {cfg.output_dir}")
        print(f"Saved merged model (for evaluation) to: {merged_output_dir}")
        print(f"Note: Use '{merged_output_dir}' in models_config.json for evaluation")
    else:
        # For full fine-tuning, save normally
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"Saved model to: {cfg.output_dir}")


if __name__ == "__main__":
    main()

