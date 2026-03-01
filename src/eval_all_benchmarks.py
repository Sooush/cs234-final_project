"""
Comprehensive evaluation script for all models on all benchmarks.
Based on the SRL paper parameters and methodology.

Evaluates:
- Models: Base, SFT, SRL, SRL→RLVR
- Benchmarks: AMC23, AIME24, AIME25, Minerva Math (mathematical reasoning)
- Modes: greedy, avg1, avg32

Uses paper parameters:
- max_gen_toks: 4096
- temperature: 1.0 (sampling), 0.0 (greedy)
- batch_size: 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import time


def run_lm_eval(
    model: str,
    model_path: str | None = None,
    backend: str = "hf",
    tasks: str = "aime24",
    batch_size: int = 1,
    temperature: float = 1.0,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    max_gen_toks: int = 4096,
    output_path: str | None = None,
    device: str = "cuda",
    limit: int | None = None,
    cache_dir: str | None = None,
) -> str:
    """
    Run lm_eval. Returns output directory path.
    Based on paper parameters: max_gen_toks=4096.
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        n_str = f"avg{num_return_sequences}" if do_sample and num_return_sequences > 1 else "greedy"
        output_path = f"results/{tasks}_{n_str}_{timestamp}"

    Path(output_path).mkdir(parents=True, exist_ok=True)

    if backend == "hf":
        model_arg = model_path if model_path else model
        # Resolve local checkpoint paths to absolute so HuggingFace loads from disk (not Hub)
        if model_arg.startswith(("checkpoints/", "checkpoints\\", "./", ".\\")) or (model_arg.startswith("checkpoints") and len(Path(model_arg).parts) > 1):
            model_arg = str(Path(model_arg).resolve())
        else:
            p = Path(model_arg)
            if p.exists() and p.is_dir():
                model_arg = str(p.resolve())
        is_local = Path(model_arg).is_absolute() or model_arg.startswith(("checkpoints", "./", ".\\"))
        model_args_str = f"pretrained={model_arg},trust_remote_code=True"
        if is_local:
            model_args_str += ",local_files_only=True"
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args_str,
            "--tasks", tasks,
            "--device", device,
            "--batch_size", str(batch_size),
            "--output_path", output_path,
            "--log_samples",
        ]
        # Use num_return_sequences: HF model.generate() does not accept 'n'
        gen_kwargs = f"temperature={temperature},do_sample={do_sample},num_return_sequences={num_return_sequences},max_gen_toks={max_gen_toks}"
    elif backend == "vllm":
        base_url = "http://127.0.0.1:8000/v1/completions"
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-completions",
            "--model_args", f"model={model},base_url={base_url},tokenized_requests=False,trust_remote_code=True",
            "--tasks", tasks,
            "--device", device,
            "--batch_size", str(batch_size),
            "--output_path", output_path,
            "--log_samples",
        ]
        gen_kwargs = f"temperature={temperature},do_sample={do_sample},n={num_return_sequences},max_gen_toks={max_gen_toks}"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Enable lm_eval caching so interrupted runs can resume without
    # re-evaluating already-completed samples.
    # Per lm-evaluation-harness docs, this uses: --use_cache <DIR>
    if cache_dir is not None:
        cmd.extend(["--use_cache", cache_dir])

    cmd.extend(["--gen_kwargs", gen_kwargs])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    with open(Path(output_path) / "eval_cmd.txt", "w") as f:
        f.write(" ".join(cmd) + "\n")

    subprocess.run(cmd, check=True)
    return output_path


def evaluate_single_config(
    model_name: str,
    model_path: Optional[str],
    benchmark: str,
    mode: str,
    device: str = "cuda",
    backend: str = "hf",
    max_gen_toks: int = 4096,
    limit: Optional[int] = None,
    results_dir: str = "results",
    cache_dir: Optional[str] = None,
) -> dict:
    """
    Evaluate a single model/benchmark/mode combination.
    Returns dict with evaluation metadata.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name} | {benchmark} | {mode}")
    print(f"{'='*80}")
    
    # Set evaluation parameters based on mode (paper parameters)
    if mode == "greedy":
        temperature = 0.0
        do_sample = False
        n = 1
    elif mode == "avg1":
        temperature = 1.0
        do_sample = True
        n = 1
    elif mode == "avg32":
        temperature = 1.0
        do_sample = True
        n = 32
        # Warn about HF backend issues with avg32
        if backend == "hf":
            print("WARNING: avg32 mode with HF backend may have compatibility issues.")
            print("Consider using --backend vllm for avg32 mode, or use avg1/greedy modes.")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create output directory name
    model_safe = model_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{results_dir}/{benchmark}_{model_safe}_{mode}_{timestamp}"
    
    start_time = time.time()
    
    try:
        output_path = run_lm_eval(
            model=model_name,
            model_path=model_path,
            backend=backend,
            tasks=benchmark,
            batch_size=1,  # Paper uses batch_size=1
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=n,
            max_gen_toks=max_gen_toks,  # Paper parameter: 4096
            output_path=output_dir,
            device=device,
            limit=limit,
            cache_dir=cache_dir,
        )
        
        elapsed_time = time.time() - start_time
        
        # Try to read results
        results_file = Path(output_path) / "results.json"
        results = {}
        if results_file.exists():
            with open(results_file, "r") as f:
                results = json.load(f)
        
        return {
            "model_name": model_name,
            "model_path": model_path,
            "benchmark": benchmark,
            "mode": mode,
            "output_path": output_path,
            "elapsed_time": elapsed_time,
            "success": True,
            "results": results,
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"ERROR: Evaluation failed: {e}")
        return {
            "model_name": model_name,
            "model_path": model_path,
            "benchmark": benchmark,
            "mode": mode,
            "output_path": output_dir,
            "elapsed_time": elapsed_time,
            "success": False,
            "error": str(e),
        }


def load_model_config(config_path: str) -> tuple[dict, str]:
    """Load model configuration from JSON file.
    Returns (model_config_dict, base_model_name).
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        # Extract base_model if present
        base_model = config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
        print(f"Using base model from config: {base_model}")
        # Handle both formats: {"models": {...}} and direct model dict
        if "models" in config:
            model_config = config["models"]
        else:
            model_config = config
        
        # Fill in model_name if it's null (use base_model)
        for model_key, model_data in model_config.items():
            if isinstance(model_data, dict) and model_data.get("model_name") is None:
                model_data["model_name"] = base_model
        
        return model_config, base_model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all models on all benchmarks using paper parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models on AIME24 with greedy decoding
  python -m src.eval_all_benchmarks --benchmarks aime24 --modes greedy

  # Evaluate specific models on all benchmarks
  python -m src.eval_all_benchmarks --models base srl --benchmarks aime24

  # Use custom model config file
  python -m src.eval_all_benchmarks --config configs/models.json

  # Quick test with limit
  python -m src.eval_all_benchmarks --limit 10
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file with model paths (default: uses built-in config)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["base", "sft", "srl", "srl_rlvr"],
        default=["base", "sft", "srl", "srl_rlvr"],
        help="Models to evaluate (default: all)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["amc23", "aime24", "aime25", "minerva_math"],
        default=["amc23", "aime24", "aime25", "minerva_math"],
        help="Benchmarks to evaluate (default: all math benchmarks)"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["greedy", "avg1", "avg32"],
        default=["greedy", "avg1", "avg32"],
        help="Evaluation modes (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "vllm"],
        help="Backend to use (default: hf)"
    )
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=4096,
        help="Max tokens to generate (paper parameter: 4096, default: 4096)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per benchmark (for quick tests)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base directory for results (default: results)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip evaluations if output directory already exists"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to use for lm_eval --use_cache <DIR> (enables caching when set)"
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path to a JSON checkpoint file to resume evaluations"
    )
    
    args = parser.parse_args()
    
    # Default base model (can be overridden by --config)
    DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    
    # Default model configuration (can be overridden by --config)
    default_model_config = {
        "base": {
            "model_name": DEFAULT_BASE_MODEL,
            "model_path": None,
        },
        "sft": {
            "model_name": DEFAULT_BASE_MODEL,
            "model_path": "checkpoints/sft/step_final",  # Update with actual SFT checkpoint
        },
        "srl": {
            "model_name": DEFAULT_BASE_MODEL,
            "model_path": "checkpoints/srl/step_500",  # Update with actual SRL checkpoint
        },
        "srl_rlvr": {
            "model_name": DEFAULT_BASE_MODEL,
            "model_path": "checkpoints/srl_rlvr/step_final",  # Update with actual SRL→RLVR checkpoint
        },
    }
    
    # Load custom config if provided
    if args.config:
        model_config, base_model = load_model_config(args.config)
        print(f"Using base model from config: {base_model}")
    else:
        model_config = default_model_config
        base_model = DEFAULT_BASE_MODEL
    
    # Validate model paths exist (warn if not)
    for model_key in args.models:
        if model_key not in model_config:
            print(f"WARNING: Model '{model_key}' not found in config, skipping")
            continue
        
        config = model_config[model_key]
        model_path = config.get("model_path")
        if model_path and not Path(model_path).exists():
            print(f"WARNING: Model path '{model_path}' does not exist. Will try to load anyway.")
    
    # Create / load results summary & checkpoint
    all_results: list[dict] = []
    completed_keys: set[str] = set()

    # Always create a fresh evaluation_summary_....json for analysis
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    summary_file = (
        Path(args.results_dir)
        / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Optional separate checkpoint file for resuming evaluations
    checkpoint_path: Optional[Path] = None
    print(f"Checkpoint file: {args.checkpoint_file}")
    if args.checkpoint_file:
        checkpoint_path = Path(args.checkpoint_file).expanduser()
        # Try to resolve to absolute path
        try:
            checkpoint_path = checkpoint_path.resolve()
        except Exception:
            # If resolution fails (e.g., parent doesn't exist), use as-is
            pass
        
        print(f"\n{'='*80}")
        print(f"Checkpoint Configuration")
        print(f"{'='*80}")
        print(f"Checkpoint file path: {checkpoint_path}")
        print(f"Absolute path: {checkpoint_path.absolute()}")
        print(f"File exists: {checkpoint_path.exists()}")
        
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint file not found. Starting fresh evaluation.")
            print(f"Expected location: {checkpoint_path.absolute()}")
            print(f"{'='*80}\n")
        else:
            # Ensure parent directory exists (for writing updated checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    saved = json.load(f)
                all_results = saved.get("results", []) or []
                
                # Use completed_keys directly from checkpoint if available
                saved_completed_keys = saved.get("completed_keys", [])
                if saved_completed_keys:
                    completed_keys = set(saved_completed_keys)
                    print(
                        f"Loaded checkpoint from {checkpoint_path}: "
                        f"{len(completed_keys)} completed evaluations from completed_keys field."
                    )
                else:
                    # Fallback: reconstruct completed keys from results
                    for r in all_results:
                        mk = r.get("model_key")
                        bm = r.get("benchmark")
                        md = r.get("mode")
                        if mk is None or bm is None or md is None:
                            continue
                        if r.get("success"):
                            completed_keys.add(f"{mk}::{bm}::{md}")
                    print(
                        f"Loaded checkpoint from {checkpoint_path}: "
                        f"{len(completed_keys)} completed evaluations (reconstructed from results)."
                    )
                
                # Debug: print what keys were loaded
                if completed_keys:
                    print(f"Completed keys: {sorted(completed_keys)}")
                print(f"{'='*80}\n")
            except Exception as e:
                print(f"WARNING: Failed to load checkpoint '{checkpoint_path}': {e}")
                import traceback
                traceback.print_exc()
    
    total_evaluations = len(args.models) * len(args.benchmarks) * len(args.modes)
    current_eval = 0
    
    print(f"\n{'='*80}")
    print(f"Starting Comprehensive Evaluation")
    print(f"{'='*80}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Device: {args.device}, Backend: {args.backend}")
    print(f"Max gen tokens: {args.max_gen_toks}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Run all evaluations
    for model_key in args.models:
        if model_key not in model_config:
            continue
        
        config = model_config[model_key]
        model_name = config["model_name"]
        model_path = config.get("model_path")
        print(f"Using model: {model_name} from path: {model_path}")
        
        for benchmark in args.benchmarks:
            for mode in args.modes:
                current_eval += 1
                print(f"\n[{current_eval}/{total_evaluations}] ", end="")

                eval_key = f"{model_key}::{benchmark}::{mode}"

                # Check checkpoint first (only skip if we previously saw a successful run)
                if eval_key in completed_keys:
                    print(f"Skipping (checkpoint): {model_key} | {benchmark} | {mode}")
                    continue
                
                # Check if should skip
                if args.skip_existing:
                    model_safe = model_name.replace("/", "_").replace("\\", "_")
                    pattern = f"{benchmark}_{model_safe}_{mode}_*"
                    existing_done = False
                    for path in Path(args.results_dir).glob(pattern):
                        results_file = path / "results.json"
                        if results_file.exists():
                            existing_done = True
                            break
                    if existing_done:
                        print(f"Skipping (exists): {model_key} | {benchmark} | {mode}")
                        continue
                
                result = evaluate_single_config(
                    model_name=model_name,
                    model_path=model_path,
                    benchmark=benchmark,
                    mode=mode,
                    device=args.device,
                    backend=args.backend,
                    max_gen_toks=args.max_gen_toks,
                    limit=args.limit,
                    results_dir=args.results_dir,
                    cache_dir=args.cache_dir,
                )

                result["model_key"] = model_key
                all_results.append(result)
                if result.get("success"):
                    completed_keys.add(eval_key)

                # Save intermediate results / checkpoint
                payload = {
                    "config": vars(args),
                    "model_config": model_config,
                    "results": all_results,
                    "completed_keys": sorted(list(completed_keys)),
                    "total_time": time.time() - start_time,
                }

                # Analysis summary (timestamped file in results_dir)
                with open(summary_file, "w") as f:
                    json.dump(payload, f, indent=2)

                # Optional separate checkpoint file for resuming
                if checkpoint_path is not None:
                    with open(checkpoint_path, "w") as f:
                        json.dump(payload, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Evaluation Complete")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
    print(f"Successful: {sum(1 for r in all_results if r.get('success'))}/{len(all_results)}")
    print(f"Failed: {sum(1 for r in all_results if not r.get('success'))}/{len(all_results)}")
    print(f"\nResults summary saved to: {summary_file}")
    print(f"{'='*80}\n")
    
    # Print results table
    print("Results Summary:")
    print("-" * 80)
    print(f"{'Model':<15} {'Benchmark':<10} {'Mode':<10} {'Status':<10} {'Time (s)':<10}")
    print("-" * 80)
    for result in all_results:
        status = "✓" if result.get("success") else "✗"
        time_str = f"{result.get('elapsed_time', 0):.1f}"
        print(f"{result.get('model_key', 'N/A'):<15} {result.get('benchmark', 'N/A'):<10} {result.get('mode', 'N/A'):<10} {status:<10} {time_str:<10}")
    print("-" * 80)


if __name__ == "__main__":
    main()

