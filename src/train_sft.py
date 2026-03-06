"""
Supervised Fine-Tuning (SFT) training script with LoRA.

Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

- learning_rate = 5e-6
- global batch size ≈ 64
- scheduler = cosine
- warmup_ratio = 0.3
- num_train_epochs = 3
- dtype = bf16 (if available)
- LoRA rank = 16 (default)
- LoRA alpha = 32 (default)

This script is intended to be run as:

    python -m src.train_sft \
      --output-dir checkpoints/sft

After training, point `configs/models_config.json["models"]["sft"]["model_path"]`
to the final checkpoint (e.g. `checkpoints/sft` or `checkpoints/sft/checkpoint-XXXX`)
so that the existing evaluation pipeline can pick it up.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

from .model_config import get_base_model


DEFAULT_DATASET = "simplescaling/s1K-1.1"


def build_chat_example(example: Dict) -> str:
    """
    Turn a s1K-style example into a chat-style training text.
    Assumes fields: "problem" and "solution" (as in s1K-1.1).
    """
    problem = example.get("problem") or example.get("question") or ""
    solution = example.get("solution") or ""

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You should think step-by-step.",
        },
        {
            "role": "user",
            "content": problem,
        },
        {
            "role": "assistant",
            "content": solution,
        },
    ]

    # The actual chat template will be applied via tokenizer.apply_chat_template
    # in the mapping function below.
    return messages  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) training")
    default_model = get_base_model()
    print(f"Default model: {default_model}")

    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Base model name or path (default: base_model from configs/models_config.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/sft",
        help="Directory to save SFT checkpoints (default: checkpoints/sft)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET,
        help="HF dataset name for SFT (default: simplescaling/s1K-1.1)",
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
        default=512,
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

    args = parser.parse_args()

    # 1. Load tokenizer first to get response template
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset
    dataset = load_dataset(args.dataset_name, split="train")
    #raw_eval = load_dataset(args.dataset_name, split="test", streaming=False)

    EVAL_SIZE=60
    TRAIN_SIZE=len(dataset) - EVAL_SIZE

    raw_train = dataset.select(range(TRAIN_SIZE))
    raw_eval = dataset.select(range(TRAIN_SIZE, (TRAIN_SIZE + EVAL_SIZE)))

    if args.max_train_samples is not None:
        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
    if args.max_eval_samples is not None:
        raw_eval = raw_eval.select(range(min(args.max_eval_samples, len(raw_eval))))

    def formatting_func(example: Dict) -> Dict[str, str]:
        messages = build_chat_example(example)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = raw_train.map(formatting_func)
    eval_dataset = raw_eval.map(formatting_func)

    # 3. Load model and apply LoRA (if enabled)
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print("model", model)

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

    # 4. Configure SFTConfig with paper hyperparameters (Table 5)
    # Paper parameters:
    # - learning_rate = 5e-6
    # - global batch size ≈ 64
    # - scheduler = cosine
    # - warmup_ratio = 0.3
    # - num_train_epochs = 3
    # - dtype = bf16 (if available)
    
    # Calculate batch size for global batch ≈ 64
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 64  # Global batch size = 64
    
 
    training_config = SFTConfig(
        # Paper hyperparameters (Table 5)
        learning_rate=5e-6,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.3,
        
        # Output and logging
        output_dir=args.output_dir,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        
        # Data type (paper uses bf16)
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        
        # SFT-specific parameters (TRL v0.28.0)
        # completion_only_loss=True masks non-assistant tokens
        completion_only_loss=True,
        
        # Sequence length (max_seq_length might be in SFTConfig in v0.28.0)
        # Try without it first, or check if it's a different parameter name
    )

    # 5. Initialize SFT trainer with SFTConfig
    # Note: Pass the model object directly (not model name) when using LoRA
    trainer = SFTTrainer(
        model=model,  # Use the LoRA-enabled model
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 6. Train
    trainer.train()

    # 7. Save final model (useful path for evaluation pipeline)
    if not args.no_lora:
        # For LoRA, save the adapter weights
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Also merge and save the full model (for easier evaluation)
        # This creates a model that can be loaded without PEFT
        merged_model = model.merge_and_unload()
        merged_output_dir = args.output_dir + "_merged"
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        print(f"Saved LoRA adapter to: {args.output_dir}")
        print(f"Saved merged model (for evaluation) to: {merged_output_dir}")
        print(f"Note: Use '{merged_output_dir}' in models_config.json for evaluation")
    else:
        # For full fine-tuning, save normally
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved model to: {args.output_dir}")


if __name__ == "__main__":
    main()

