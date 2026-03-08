"""
Aggregate and summarize evaluation results from multiple benchmark runs.
Creates a comprehensive report comparing all models across all benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_results_json(results_path: Path) -> Dict[str, Any]:
    """Load results from lm-eval results.json file."""
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {results_path}: {e}")
        return {}


def extract_metric(results: Dict[str, Any], task: str, metric: str = "acc") -> float | None:
    """
    Extract metric from lm-eval results.
    Results structure: {task: {metric: value, ...}, ...}
    """
    # lm-eval stores metrics under a top-level "results" key:
    # {"results": {task: {...}}, ...}
    results_root = results.get("results", results)

    if task not in results_root:
        return None
    
    task_results = results_root[task]
    if isinstance(task_results, dict):
        # Try common metric names
        for m in [metric, f"{task}_{metric}", "acc", "accuracy", "exact_match"]:
            if m in task_results:
                value = task_results[m]
                if isinstance(value, dict) and "acc" in value:
                    return value["acc"]
                elif isinstance(value, (int, float)):
                    return float(value)
        # If no metric found, try to get first numeric value
        for key, value in task_results.items():
            if isinstance(value, (int, float)):
                return float(value)
    
    return None


def aggregate_from_results_dir(top_dir: Path) -> pd.DataFrame:
    """
    Traverse all subfolders under top_dir, find every evaluation_summary_*.json
    (in any nested subfolder), merge their results entries, and build an aggregated
    DataFrame using model_key and mode from each summary.
    """
    top_dir = top_dir.resolve()
    if not top_dir.is_dir():
        print(f"Warning: {top_dir} is not a directory.")
        return pd.DataFrame()

    # Recursively find evaluation_summary_*.json in every subfolder
    summary_files = []
    for root, _dirs, files in os.walk(top_dir):
        root_path = Path(root)
        for name in files:
            if name.startswith("evaluation_summary_") and name.endswith(".json"):
                summary_files.append(root_path / name)
    summary_files = sorted(summary_files)

    if not summary_files:
        print(f"No evaluation_summary_*.json files found under {top_dir} (searched all subfolders).")
        return pd.DataFrame()

    print(f"Found {len(summary_files)} evaluation summary file(s) in subfolders.")

    all_results = []
    for summary_path in summary_files:
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
            for r in data.get("results", []):
                r["_summary_dir"] = str(summary_path.parent)
            all_results.extend(data.get("results", []))
        except Exception as e:
            print(f"Warning: Could not load {summary_path}: {e}")
            continue

    if not all_results:
        print("No 'results' entries in any summary file.")
        return pd.DataFrame()

    rows = []
    for result in all_results:
        if not result.get("success"):
            continue

        output_path = Path(result["output_path"])
        summary_dir = Path(result.get("_summary_dir", ""))
        benchmark = result["benchmark"]
        model_key = result.get("model_key", "unknown")
        mode = result.get("mode", "unknown")

        results_file = output_path / "results.json"
        if not results_file.exists():
            # Search under output_path first
            candidates = sorted(output_path.rglob("results_*.json")) + sorted(
                output_path.rglob("results.json")
            )
            # If output_path doesn't exist (e.g. path from another machine), search under the summary file's directory
            if not candidates and summary_dir:
                candidates = sorted(Path(summary_dir).rglob("results_*.json")) + sorted(
                    Path(summary_dir).rglob("results.json")
                )
            candidates = [p for p in candidates if p.exists() and p.is_file()]
            seen = set()
            unique = []
            for p in candidates:
                k = p.resolve()
                if k not in seen:
                    seen.add(k)
                    unique.append(p)
            candidates = unique
            if not candidates:
                continue
            results_file = None
            for cand in candidates:
                data_try = load_results_json(cand)
                if extract_metric(data_try, benchmark, "acc") is not None:
                    results_file = cand
                    break
            if results_file is None:
                results_file = candidates[0]

        results_data = load_results_json(results_file)
        metric_value = extract_metric(results_data, benchmark, "acc")

        rows.append({
            "model": model_key,
            "benchmark": benchmark,
            "mode": mode,
            "metric": metric_value,
            "elapsed_time": result.get("elapsed_time", 0),
            "output_path": str(output_path),
        })
    return pd.DataFrame(rows)


def aggregate_evaluation_summary(summary_file: Path) -> pd.DataFrame:
    """Load evaluation summary JSON and create aggregated DataFrame."""
    with open(summary_file, "r") as f:
        data = json.load(f)
    
    rows = []
    for result in data.get("results", []):
        if not result.get("success"):
            continue
        
        output_path = Path(result["output_path"])
        benchmark = result["benchmark"]
        results_file = output_path / "results.json"
        if not results_file.exists():
            # Search all subfolders of output_path for results_*.json and results.json.
            candidates = sorted(output_path.rglob("results_*.json")) + sorted(
                output_path.rglob("results.json")
            )
            candidates = [p for p in candidates if p.exists() and p.is_file()]
            seen = set()
            unique = []
            for p in candidates:
                k = p.resolve()
                if k not in seen:
                    seen.add(k)
                    unique.append(p)
            candidates = unique
            if not candidates:
                continue
            # Use first candidate that yields a valid metric for this benchmark, else first file.
            results_file = None
            for cand in candidates:
                data_try = load_results_json(cand)
                if extract_metric(data_try, benchmark, "acc") is not None:
                    results_file = cand
                    break
            if results_file is None:
                results_file = candidates[0]
        
        results_data = load_results_json(results_file)
        
        metric_value = extract_metric(results_data, benchmark, "acc")
        
        rows.append({
            "model": result.get("model_key", "unknown"),
            "benchmark": benchmark,
            "mode": result["mode"],
            "metric": metric_value,
            "elapsed_time": result.get("elapsed_time", 0),
            "output_path": str(output_path),
        })
    
    return pd.DataFrame(rows)


def create_summary_table(df: pd.DataFrame) -> str:
    """Create a formatted summary table matching the requested format."""
    if df.empty:
        return "No results to display."

    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("Evaluation Results Summary")
    summary_lines.append("=" * 100)
    summary_lines.append("")

    for benchmark in sorted(df["benchmark"].unique()):
        summary_lines.append(f"Benchmark: {benchmark.upper()}")
        summary_lines.append("-" * 100)
        summary_lines.append("")

        for mode in ["greedy", "avg1", "avg32"]:
            mode_df = df[(df["benchmark"] == benchmark) & (df["mode"] == mode)]
            if mode_df.empty:
                continue

            summary_lines.append(f"  Mode: {mode}")
            summary_lines.append("  Model                Metric          Time (s)       ")
            summary_lines.append("  --------------------------------------------------")

            for _, row in mode_df.iterrows():
                metric_str = f"{row['metric']:.4f}" if row["metric"] is not None else "N/A"
                summary_lines.append(
                    f"  {row['model']:<20} {metric_str:<15} {row['elapsed_time']:.1f}"
                )
            summary_lines.append("")

    summary_lines.append("=" * 100)
    return "\n".join(summary_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results from benchmark runs"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to an evaluation summary JSON file, or to a top folder to scan for results_*.json / results.json in all subfolders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base path for aggregated results (default: path stem + '_aggregated_results')",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "txt", "json", "all"],
        default="all",
        help="Output format (default: all)",
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}")
        return
    
    if path.is_dir():
        print(f"Scanning directory for evaluation_summary_*.json: {path}")
        df = aggregate_from_results_dir(path)
        base_path = path / path.name if path.name else path / "aggregated"
    else:
        print(f"Loading summary from: {path}")
        df = aggregate_evaluation_summary(path)
        base_path = path.with_suffix("")
    
    if df.empty:
        print("Warning: No successful evaluations found.")
        return
    
    print(f"\nLoaded {len(df)} evaluation row(s)")
    
    # Determine output paths (base name + "_aggregated_results")
    if args.output:
        base_path = Path(args.output)
    aggregated_base = base_path.parent / (base_path.stem + "_aggregated_results")
    
    # Generate outputs
    if args.format in ["csv", "all"]:
        csv_path = aggregated_base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
    
    if args.format in ["txt", "all"]:
        txt_path = aggregated_base.with_suffix(".txt")
        summary_table = create_summary_table(df)
        with open(txt_path, "w") as f:
            f.write(summary_table)
        print(f"Text summary saved to: {txt_path}")
        print("\n" + summary_table)
    
    if args.format in ["json", "all"]:
        json_path = aggregated_base.with_suffix(".json")
        df_dict = df.to_dict(orient="records")
        with open(json_path, "w") as f:
            json.dump(df_dict, f, indent=2)
        print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
