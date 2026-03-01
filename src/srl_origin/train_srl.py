"""
SRL training entrypoint with LoRA: load config, prepare model, run GRPO training with resume support.

Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .grpo_trainer import GRPOTrainer
from .utils import set_seed


def get_device() -> torch.device:
    """Return best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(output_dir: str) -> Path | None:
    """
    Find the latest checkpoint under output_dir by step number.
    Looks for subdirs named step_N (and optionally 'best'), reads trainer_step.txt
    or parses step from dir name, and returns the path with the highest step.
    Returns None if no valid checkpoint is found.
    """
    out = Path(output_dir)
    if not out.exists() or not out.is_dir():
        return None
    best_path: Path | None = None
    best_step = -1
    for d in out.iterdir():
        if not d.is_dir():
            continue
        step = -1
        step_file = d / "trainer_step.txt"
        if step_file.exists():
            try:
                step = int(step_file.read_text().strip())
            except (ValueError, OSError):
                pass
        if step < 0 and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_", 1)[1])
            except ValueError:
                pass
        # Must look like a checkpoint: has adapter or config
        has_adapter = (d / "adapter_config.json").exists() or list(d.glob("adapter_model*.bin")) or list(d.glob("adapter_model*.safetensors"))
        has_config = (d / "config.json").exists()
        if step >= 0 and (has_adapter or has_config) and step > best_step:
            best_step = step
            best_path = d
    return best_path


def main():
    parser = argparse.ArgumentParser(description="SRL GRPO training")
    parser.add_argument("--config", type=str, default=None, help="YAML config path (overrides other args)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
    parser.add_argument(
        "--init-from",
        type=str,
        default="",
        help=(
            "If this directory exists and --resume is not set, initialize SRL "
            "from this checkpoint instead of the base model (default: checkpoints/sft). "
            "Set to empty string ('') to disable and use the base model directly."
        ),
    )
    parser.add_argument("--data", type=str, default="data/srl_instances.jsonl", help="SRL instances JSONL")
    parser.add_argument("--output-dir", type=str, default="checkpoints/srl", help="Checkpoint output dir")
    parser.add_argument("--num-steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (prompts)")
    parser.add_argument("--group-size", type=int, default=4, help="Rollouts per prompt (G)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="GRPO clip epsilon")
    parser.add_argument("--eps-std", type=float, default=1e-4, help="Min reward std for dynamic filter")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest checkpoint in --output-dir (finds highest step_* or best)",
    )
    parser.add_argument("--val-data", type=str, default=None, help="Validation JSONL path for best-checkpoint selection")
    parser.add_argument("--eval-every", type=int, default=50, help="Validate every N steps")
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

    if args.config:
        cfg = load_config(args.config)
        mapping = {
            "model": "model", "data": "data", "output_dir": "output_dir",
            "num_steps": "num_steps", "batch_size": "batch_size", "group_size": "group_size",
            "max_new_tokens": "max_new_tokens", "temperature": "temperature",
            "lr": "lr", "clip_epsilon": "clip_epsilon", "eps_std": "eps_std",
            "kl_coef": "kl_coef", "checkpoint_every": "checkpoint_every", "seed": "seed",
            "resume": "resume", "resume_latest": "resume_latest", "val_data": "val_data", "eval_every": "eval_every",
            "init_from": "init_from",
        }
        for k, v in cfg.items():
            attr = mapping.get(k, k)
            if hasattr(args, attr):
                setattr(args, attr, v)

    # If --resume-latest, find latest checkpoint in output_dir and set resume
    if getattr(args, "resume_latest", False):
        latest = find_latest_checkpoint(args.output_dir)
        if latest is None:
            args.resume = None
            print(f"No checkpoint found under output_dir={args.output_dir}. Running from scratch.")
        else:
            args.resume = str(latest)
            print(f"Resume from latest checkpoint: {args.resume}")

    set_seed(args.seed)
    device = get_device()

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        resume_path = Path(args.resume)
        
        # Check if this is a LoRA checkpoint (has adapter_config.json)
        is_lora_checkpoint = (resume_path / "adapter_config.json").exists()
        
        if is_lora_checkpoint and not args.no_lora:
            # Load base model first, then load LoRA adapter
            # Try to find base model path or use the model argument
            print("Detected LoRA checkpoint, loading base model and adapter...")
            base_model_path = args.model  # Default to base model
            # Check if there's a merged model or base model reference
            merged_path = resume_path.parent / "merged"
            if merged_path.exists():
                base_model_path = str(merged_path)
                print(f"Found merged model at {base_model_path}, using as base")
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            # Load LoRA adapter
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(resume_path))
            print("Loaded LoRA adapter from checkpoint")
        else:
            # Load regular checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                args.resume, 
                dtype=torch.bfloat16, 
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        
        tokenizer = AutoTokenizer.from_pretrained(args.resume, trust_remote_code=True)
        step_file = resume_path / "trainer_step.txt"
        if step_file.exists():
            start_step = int(step_file.read_text().strip())
            print(f"Resuming from step {start_step}, {args.num_steps - start_step} steps remaining")
    else:
        init_from_path = args.init_from.strip() if args.init_from else ""
        init_path = Path(init_from_path) if init_from_path else None
        
        # Check if checkpoint directory exists and has valid files
        has_valid_checkpoint = False
        is_lora_checkpoint = False
        
        # If --init-from is empty, skip checkpoint loading and use base model
        if not init_from_path:
            has_valid_checkpoint = False
        elif init_path and init_path.exists():
            # Check if this is a LoRA checkpoint (has adapter_config.json and adapter files)
            adapter_config = init_path / "adapter_config.json"
            adapter_model = list(init_path.glob("adapter_model*.bin")) + list(init_path.glob("adapter_model*.safetensors"))
            is_lora_checkpoint = adapter_config.exists() and len(adapter_model) > 0
            
            # Check if this is a regular checkpoint (has config.json)
            has_config = (init_path / "config.json").exists()
            
            # Also check for merged model
            merged_path = init_path.parent / (init_path.name + "_merged")
            has_merged_model = merged_path.exists() and (merged_path / "config.json").exists()
            
            has_valid_checkpoint = is_lora_checkpoint or has_config or has_merged_model
        
        if has_valid_checkpoint:
            print(f"Initializing SRL from SFT checkpoint: {init_from_path}")
            
            if is_lora_checkpoint and not args.no_lora:
                # Load base model first, then load LoRA adapter
                print("Detected LoRA checkpoint, loading base model and adapter...")
                # Try to find merged model first
                merged_path = init_path.parent / (init_path.name + "_merged")
                if merged_path.exists() and (merged_path / "config.json").exists():
                    print(f"Found merged model at {merged_path}, using as base")
                    model = AutoModelForCausalLM.from_pretrained(
                        str(merged_path),
                        dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map="auto" if torch.cuda.is_available() else None,
                    )
                else:
                    # Load base model and then LoRA adapter
                    from .model_config import get_base_model
                    base_model = get_base_model()
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map="auto" if torch.cuda.is_available() else None,
                    )
                    # Load LoRA adapter
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, str(init_path))
                    print("Loaded LoRA adapter from SFT checkpoint")
                
                # Try to load tokenizer from checkpoint, fallback to base model if not found
                try:
                    tokenizer = AutoTokenizer.from_pretrained(init_from_path, trust_remote_code=True)
                except:
                    from .model_config import get_base_model
                    base_model = get_base_model()
                    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            else:
                # Load regular checkpoint
                model = AutoModelForCausalLM.from_pretrained(
                    init_from_path,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                tokenizer = AutoTokenizer.from_pretrained(init_from_path, trust_remote_code=True)
        else:
            # No valid checkpoint found, use base model
            from .model_config import get_base_model
            base_model = get_base_model()
            if not init_from_path:
                print(f"--init-from is empty; using base model: {base_model}")
            elif init_path and init_path.exists():
                print(f"Init-from path '{init_from_path}' exists but contains no valid checkpoint files; falling back to base model: {base_model}")
            else:
                print(f"Init-from path '{init_from_path}' not found; falling back to base model: {base_model}")
            
            # Load base model (LoRA will be applied later if enabled)
            print(f"Loading base model: {base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check if model already has LoRA applied (from checkpoint)
    # Check if model is a PEFT model by checking for peft_config attribute
    from peft import PeftModel
    model_has_lora = isinstance(model, PeftModel) or hasattr(model, 'peft_config')
    print("model", model)
    print("model_has_lora", model_has_lora)

    # Apply LoRA if enabled and model doesn't already have LoRA
    if not args.no_lora and not model_has_lora:
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
        # For full fine-tuning, ensure all parameters require gradients
        for param in model.parameters():
            param.requires_grad = True

    # Ensure model is in training mode and parameters require gradients
    model.train()
    print("Disabling use_cache and enabling gradient checkpointing")
    model.use_cache = False
    model.gradient_checkpointing_enable()
    
    # Verify that some parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found! Check LoRA configuration or model setup.")
    
    if not torch.cuda.is_available() or device.type == "cpu":
        model = model.to(device)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        ref_model=None,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        kl_coef=0.0,
        clip_epsilon=args.clip_epsilon,
        eps_std=args.eps_std,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    trainer.train(
        data_path=args.data,
        num_steps=args.num_steps,
        device=device,
        start_step=start_step,
        val_data_path=getattr(args, "val_data", None),
        eval_every=getattr(args, "eval_every", 50),
    )
    
    # Save merged model if LoRA was used (for easier evaluation)
    if not args.no_lora:
        print("\nMerging and saving LoRA adapter...")
        try:
            # Merge LoRA weights into base model
            merged_model = model.merge_and_unload()
            merged_output_dir = Path(args.output_dir) / "merged"
            merged_output_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(merged_output_dir)
            tokenizer.save_pretrained(merged_output_dir)
            print(f"Saved merged model (for evaluation) to: {merged_output_dir}")
            print(f"Note: Use '{merged_output_dir}' in models_config.json for evaluation")
        except Exception as e:
            print(f"Warning: Could not merge LoRA adapter: {e}")
            print("LoRA adapter weights are saved in the checkpoint directory")
    
    print("Training complete.")


if __name__ == "__main__":
    main()
