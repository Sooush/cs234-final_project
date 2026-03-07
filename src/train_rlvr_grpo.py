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

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig
import torch

from .model_config import get_base_model


DEFAULT_DATASET = "simplescaling/s1K-1.1"

FORMAT_REWARD_VALUE = 0.2
ACCURACY_REWARD_VALUE = 0.8


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
    max_completion_length: int = 4096  # max tokens to generate per completion
    num_train_epochs: int = 2  # paper uses steps; epochs is a practical proxy
    beta: float = 0.0  # KL coeff


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


def _has_think_format(text: str) -> bool:
    """
    True if the completion has <think>...</think> format (opening and closing think tags
    with closing tag after the opening, and some content in or after the block).
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
    """
    if prompt_builder is None:
        prompt_builder = build_prompt

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
            for i, (r, p, c) in enumerate(zip(rewards, prompts[:3], completions[:3])):
                key = (p.strip() if p else "")
                gt_raw = prompt_to_gt.get(key, "")
                gt = extract_final_answer(gt_raw)
                pred = extract_final_answer(c)
                print(f"[RLVR reward] sample {i}: gt={gt!r} pred={pred!r} reward={r} (key_in_map={key[:50]!r}...)")
            os.environ["RLVR_DEBUG_REWARD"] = "0"  # log only once

        #try:
        #    with open("/tmp/rlvr_reward.log", "a") as f:
        #        f.write(f"Accuracy Reward: {rewards[-1]}\n")
        #except OSError:
        #    pass
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
            #try:
            #    with open("/tmp/rlvr_reward.log", "a") as f:
            #        f.write(f"Format Reward: {rewards[-1]}\n Completion: {completion}\n")
            #except OSError:
            #    pass
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
        default="checkpoints/srl_rlvr",
        help="Directory to save RLVR checkpoints (default: checkpoints/srl_rlvr)",
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
        default=256,
        help="Optional: limit number of eval samples",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling parameter (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
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
        default=4096,
        help="Max tokens to generate per completion (default: 4096)",
    )
    args = parser.parse_args()

    cfg = RLVRConfig(
        init_from=args.init_from,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_completion_length=args.max_completion_length,
    )

    base_model = get_base_model()
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
    dataset = load_dataset(cfg.dataset_name, split="train").shuffle(seed=42)
    
    # Split train/val from the shuffled dataset
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
        return {
            "prompt": build_chat_prompt(example),
            "completion": example.get("solution", ""),
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
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        print(f"Applying LoRA with rank={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        print(f"Target modules: {target_modules}")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Using full fine-tuning (LoRA disabled)")

    # Enable gradient checkpointing to reduce memory usage
    print("Disabling use_cache and enabling gradient checkpointing")
    model.use_cache = False
    model.gradient_checkpointing_enable()

    # 3. Model loading config for GRPOConfig (if needed)
    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "low_cpu_mem_usage": True,
    }

    # 4. GRPO training configuration
    per_device_train_batch_size = 1
    gradient_accumulation_steps = cfg.batch_size // per_device_train_batch_size
    print(f"Per device train batch size: {per_device_train_batch_size}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # GRPOConfig's TrainingArguments
    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        logging_steps=10,  # Log every 10 steps
        logging_first_step=True,  # Log the first step
        log_level="info",  # Info level logging
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=gradient_accumulation_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=1.0,
        beta=cfg.beta,  # KL coeff
        model_init_kwargs=model_kwargs,
        report_to="none",  # No external logging (wandb/tensorboard)
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        # Evaluation and saving
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,  # Evaluate every 50 steps if eval dataset provided
        save_strategy="steps",
        save_steps=5,  # Save checkpoint every 5 steps
        save_total_limit=3,  # Keep only the last 3 checkpoints
        load_best_model_at_end=False,  # Don't load best model (we'll handle this manually if needed)
        # Logging details
        logging_dir=cfg.output_dir,  # Directory for logs
        run_name="rlvr_grpo_training",  # Name for this training run
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
    print(f"Temperature: 1.0")
    print(f"Beta (KL coeff): {cfg.beta}")
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

