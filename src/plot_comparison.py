"""
Generate comparison graphs for all models across all mathematical reasoning benchmarks.
Creates bar charts, line plots, and heatmaps comparing model performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def get_model_display_name(model_key: str) -> str:
    """
    Get display name for a model key.
    
    Args:
        model_key: Model identifier (e.g., "base", "sft", "srl", "srl_rlvr")
    
    Returns:
        Display name for the model
    """
    model_labels = {
        "base": "Qwen2.5-3B-Instruct",
        "sft": "SFT",
        "rlvr": "RLVR",
        "srl": "SRL",
        "srl_rlvr": "SRL->RLVR",
        "srl-l4-token-8192-0313v2": "SRL Fine-tuned",
        "rlvr-l4-token-8192-0314": "RLVR Fine-tuned"
    }
    return model_labels.get(model_key, model_key.upper())


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a decimal value (0.0-1.0) as a percentage string.
    
    Args:
        value: Decimal value between 0.0 and 1.0
        precision: Number of decimal places (default: 1)
    
    Returns:
        Formatted percentage string (e.g., "45.2%")
    """
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.{precision}f}%"


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
    if task not in results:
        return None
    
    task_results = results[task]
    if isinstance(task_results, dict):
        # Try common metric names
        for m in [metric, f"{task}_{metric}", "acc", "accuracy", "exact_match", "pass_at_1"]:
            if m in task_results:
                value = task_results[m]
                if isinstance(value, dict):
                    # Handle nested dicts like {"acc": 0.5, "acc_stderr": 0.01}
                    if "acc" in value:
                        return float(value["acc"])
                    elif "pass_at_1" in value:
                        return float(value["pass_at_1"])
                elif isinstance(value, (int, float)):
                    return float(value)
        # If no metric found, try to get first numeric value
        for key, value in task_results.items():
            if isinstance(value, (int, float)) and not key.endswith("_stderr"):
                return float(value)
    
    return None


def load_evaluation_summary(summary_file: Path) -> pd.DataFrame:
    """
    Load evaluation summary and create aggregated DataFrame.

    Supports two formats:
    1) Raw eval summary from eval_all_benchmarks.py:
       {"results": [...], ...}
    2) Aggregated JSON from aggregate_results.py:
       [{"model": ..., "benchmark": ..., "mode": ..., "metric": ...}, ...]
    """
    with open(summary_file, "r") as f:
        data = json.load(f)

    # Case 1: already-aggregated list of rows (from aggregate_results.py)
    if isinstance(data, list):
        return pd.DataFrame(data)

    # Case 2: raw eval summary with "results" entries
    rows = []
    for result in data.get("results", []):
        if not result.get("success"):
            continue

        output_path = Path(result["output_path"])
        results_file = output_path / "results.json"

        if not results_file.exists():
            # Fallback for lm-eval layout: look for results_*.json under subdirs
            candidates = list(output_path.rglob("results_*.json"))
            if not candidates:
                continue
            results_file = candidates[0]

        results_data = load_results_json(results_file)
        benchmark = result["benchmark"]

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


def create_bar_chart(df: pd.DataFrame, output_path: Path, mode: str = "greedy"):
    """Create bar chart comparing models across benchmarks for a specific mode."""
    mode_df = df[df["mode"] == mode].copy()
    if mode_df.empty:
        print(f"Warning: No data for mode {mode}")
        return
    
    # Pivot table: models as rows, benchmarks as columns
    pivot = mode_df.pivot_table(
        index="model",
        columns="benchmark",
        values="metric",
        aggfunc="first"
    )
    
    # Sort models in a logical order
    model_order = ["base", "sft", "rlvr", "srl", "srl_rlvr", "srl-l4-token-8192-0313v2", "rlvr-l4-token-8192-0314"]
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(pivot.index))
    #width = 0.12  # Thinner bars (reduced from 0.2)
    width = 0.02
    benchmarks = pivot.columns.tolist()
    num_benchmarks = len(benchmarks)
    
    for i, benchmark in enumerate(benchmarks):
        offset = (i - num_benchmarks / 2 + 0.5) * width
        # Convert values to percentage for display
        bars = ax.bar(x + offset, pivot[benchmark].values * 100, width, label=benchmark.upper())
        
        # Add value labels on bars (convert to percentage)
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       format_percentage(height),
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Comparison Across Benchmarks ({mode.upper()} Mode)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([get_model_display_name(m) for m in pivot.index], fontsize=10)
    ax.legend(title='Benchmark', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart to: {output_path}")


def create_line_plot(df: pd.DataFrame, output_path: Path, mode: str = "greedy"):
    """Create line plot showing model performance across benchmarks."""
    mode_df = df[df["mode"] == mode].copy()
    if mode_df.empty:
        print(f"Warning: No data for mode {mode}")
        return
    
    # Pivot table: benchmarks as x-axis, models as lines
    pivot = mode_df.pivot_table(
        index="benchmark",
        columns="model",
        values="metric",
        aggfunc="first"
    )
    
    # Sort models
    model_order = ["base", "sft", "rlvr", "srl", "srl_rlvr", "srl-l4-token-8192-0313v2", "rlvr-l4-token-8192-0314"]
    available_models = [m for m in model_order if m in pivot.columns]
    pivot = pivot[available_models]
    
    # Sort benchmarks
    benchmark_order = ["amc23", "aime24", "aime25", "minerva_math"]
    pivot = pivot.reindex([b for b in benchmark_order if b in pivot.index])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in available_models:
        # Convert values to percentage for display
        ax.plot(pivot.index, pivot[model] * 100, marker='o', linewidth=2, 
               markersize=8, label=get_model_display_name(model))
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance Across Benchmarks ({mode.upper()} Mode)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Model', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved line plot to: {output_path}")


def create_heatmap(df: pd.DataFrame, output_path: Path, mode: str = "greedy"):
    """Create heatmap showing model performance across benchmarks."""
    mode_df = df[df["mode"] == mode].copy()
    if mode_df.empty:
        print(f"Warning: No data for mode {mode}")
        return
    
    # Pivot table: models as rows, benchmarks as columns
    pivot = mode_df.pivot_table(
        index="model",
        columns="benchmark",
        values="metric",
        aggfunc="first"
    )
    
    # Sort models and benchmarks
    model_order = ["base", "sft", "rlvr", "srl", "srl_rlvr", "srl-l4-token-8192-0313v2", "rlvr-l4-token-8192-0314"]
    benchmark_order = ["amc23", "aime24", "aime25", "minerva_math"]
    
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])
    pivot = pivot[[b for b in benchmark_order if b in pivot.columns]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert values to percentage for display
    pivot_pct = pivot * 100
    sns.heatmap(pivot_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Accuracy (%)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance Heatmap ({mode.upper()} Mode)',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels([b.upper() for b in pivot.columns], rotation=45, ha='right')
    ax.set_yticklabels([get_model_display_name(m) for m in pivot.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def create_comprehensive_bar_chart(df: pd.DataFrame, output_path: Path):
    """Create a single comprehensive bar chart showing all models across all benchmarks and modes."""
    if df.empty:
        print("Warning: No data for comprehensive chart")
        return
    
    # Create a combined label: benchmark_mode
    df_combined = df.copy()
    df_combined['benchmark_mode'] = df_combined['benchmark'] + '_' + df_combined['mode']
    
    # Pivot: models as x-axis, benchmark_mode combinations as grouped bars
    pivot = df_combined.pivot_table(
        index="model",
        columns="benchmark_mode",
        values="metric",
        aggfunc="first"
    )
    
    # Sort models
    model_order = ["base", "sft", "rlvr", "srl", "srl_rlvr", "srl-l4-token-8192-0313v2", "rlvr-l4-token-8192-0314"]
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])
    
    # Sort columns by benchmark then mode
    benchmark_order = ["amc23", "aime24", "aime25", "minerva_math"]
    mode_order = ["greedy", "avg1", "avg32"]
    
    sorted_cols = []
    for bench in benchmark_order:
        for mode in mode_order:
            col = f"{bench}_{mode}"
            if col in pivot.columns:
                sorted_cols.append(col)
    # Add any remaining columns
    for col in pivot.columns:
        if col not in sorted_cols:
            sorted_cols.append(col)
    
    pivot = pivot[sorted_cols]
    
    num_models = len(pivot.index)
    num_groups = len(pivot.columns)
    
    # Auto-adjust figure width based on number of models (more models = wider figure)
    fig_width = max(12, min(28, 3.5 * num_models))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    x = np.arange(len(pivot.index))
    # Auto-adjust bar width so all groups fit without overlap: total span per tick ~0.8
    # width per group = 0.8 / num_groups, clamped so bars are visible but not huge
    width = np.clip(0.8 / max(num_groups, 1), 0.02, 0.15)
    
    if num_models == 1:
        x_center = 0
    else:
        x_center = None
    
    # Create color map for better distinction
    colors = plt.cm.tab20(np.linspace(0, 1, num_groups))
    
    for i, (col, color) in enumerate(zip(pivot.columns, colors)):
        if num_models == 1:
            offset = (i - num_groups / 2 + 0.5) * width
            bars = ax.bar(x_center + offset, pivot[col].values * 100, width, 
                         label=col.replace('_', ' ').upper(), color=color, alpha=0.8)
        else:
            offset = (i - num_groups / 2 + 0.5) * width
            bars = ax.bar(x + offset, pivot[col].values * 100, width, 
                         label=col.replace('_', ' ').upper(), color=color, alpha=0.8)
        
        # Add value labels on bars (values already in percentage from * 100)
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height >= 0:  # Include 0 so zero values are displayed
                if height < 1.0 and height > 0:
                    label = f'{height:.2f}%'
                elif height == 0:
                    label = '0%'
                else:
                    label = f'{height:.1f}%'
                y_pos = max(height, 0.5)  # Place 0% label slightly above baseline so it's visible
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       label,
                       ha='center', va='bottom', fontsize=6, rotation=90)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Comprehensive Model Comparison: All Benchmarks and Modes', 
                 fontsize=16, fontweight='bold', pad=20)
    
    if num_models == 1:
        ax.set_xticks([0])
        ax.set_xticklabels([get_model_display_name(m) for m in pivot.index], fontsize=12)
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([get_model_display_name(m) for m in pivot.index], fontsize=12)
    
    # Place legend outside the plot area
    ax.legend(title='Benchmark_Mode', bbox_to_anchor=(1.02, 1), loc='upper left', 
              fontsize=8, ncol=2, title_fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive bar chart to: {output_path}")


def create_benchmark_mode_plot(df: pd.DataFrame, output_path: Path):
    """
    Create a plot with benchmark(mode) on x-axis, benchmark name at top, 
    and model name in upper right corner.
    Format: AIME24(Avg@32), AIME25(Greedy), etc.
    """
    if df.empty:
        print("Warning: No data for benchmark-mode plot")
        return
    
    # Create combined labels: benchmark(mode)
    df_combined = df.copy()
    
    # Map mode to display format
    mode_labels = {
        "greedy": "Greedy",
        "avg1": "Avg@1",
        "avg32": "Avg@32"
    }
    
    df_combined['benchmark_mode'] = df_combined.apply(
        lambda row: f"{row['benchmark'].upper()}({mode_labels.get(row['mode'], row['mode'].upper())})",
        axis=1
    )
    
    # Pivot: benchmark_mode as x-axis, models as bars
    pivot = df_combined.pivot_table(
        index="benchmark_mode",
        columns="model",
        values="metric",
        aggfunc="first"
    )
    
    # Sort by benchmark then mode
    benchmark_order = ["amc23", "aime24", "aime25", "minerva_math"]
    mode_order = ["greedy", "avg1", "avg32"]
    
    sorted_index = []
    for bench in benchmark_order:
        for mode in mode_order:
            label = f"{bench.upper()}({mode_labels.get(mode, mode.upper())})"
            if label in pivot.index:
                sorted_index.append(label)
    
    # Add any remaining
    for idx in pivot.index:
        if idx not in sorted_index:
            sorted_index.append(idx)
    
    pivot = pivot.reindex(sorted_index)
    
    # Sort models
    model_order = ["base", "sft", "rlvr", "srl", "srl_rlvr", "srl-l4-token-8192-0313v2", "rlvr-l4-token-8192-0314"]
    available_models = [m for m in model_order if m in pivot.columns]
    pivot = pivot[available_models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(pivot.index))
    #width = 0.18  # Bar width
    width = 0.02
    num_models = len(available_models)
    
    # Colors for models
    colors = {
        "base": "#808080",      # Gray
        "sft": "#4A90E2",       # Blue
        "srl": "#50C878",       # Green
        "srl_rlvr": "#FF6B35"   # Orange/Red
    }
    
    # Plot bars for each model (convert to percentage)
    for i, model in enumerate(available_models):
        offset = (i - num_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            pivot[model].values * 100,
            width,
            label=get_model_display_name(model),
            color=colors.get(model, plt.cm.tab10(i)),
            alpha=0.85,
            edgecolor='black',
            linewidth=0.8
        )
        
        # Add value labels on bars (values already in percentage from * 100)
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height >= 0:
                if height < 1.0 and height > 0:
                    label = f'{height:.2f}%'
                elif height == 0:
                    label = '0%'
                else:
                    label = f'{height:.1f}%'
                y_pos = max(height, 0.5)  # Place 0% label slightly above baseline so it's visible
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    y_pos,
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )
    
    # Extract benchmark names for top annotation
    benchmark_names = []
    for idx in pivot.index:
        # Extract benchmark name (before the parenthesis)
        bench_name = idx.split('(')[0]
        benchmark_names.append(bench_name)
    
    # Set x-axis labels (benchmark(mode))
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=10, fontweight='bold', rotation=45, ha='right')
    
    # Add benchmark names at the top
    y_max = ax.get_ylim()[1]
    for i, (idx, bench_name) in enumerate(zip(pivot.index, benchmark_names)):
        # Only show unique benchmark names (avoid duplicates)
        if i == 0 or benchmark_names[i] != benchmark_names[i-1]:
            ax.text(
                i, y_max * 1.05,
                bench_name,
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
                color='#333333'
            )
    
    # Add model names in upper right corner
    model_text = " | ".join([get_model_display_name(m) for m in available_models])
    ax.text(
        0.98, 0.98,
        model_text,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5),
        zorder=10
    )
    
    # Labels and title
    ax.set_xlabel('Benchmark (Mode)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Across Benchmarks and Modes', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(
        title='Model',
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        loc='upper left'
    )
    
    # Grid and limits
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0, top=y_max * 1.15)  # Extra space for top labels
    
    # Background
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved benchmark-mode plot to: {output_path}")


def create_multi_mode_comparison(df: pd.DataFrame, output_path: Path):
    """Create grouped bar chart comparing all modes for each model."""
    # Group by model, benchmark, and mode
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    benchmarks = sorted(df["benchmark"].unique())
    
    for idx, benchmark in enumerate(benchmarks):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        bench_df = df[df["benchmark"] == benchmark].copy()
        
        # Pivot: models as x-axis, modes as grouped bars
        pivot = bench_df.pivot_table(
            index="model",
            columns="mode",
            values="metric",
            aggfunc="first"
        )
        
        model_order = ["base", "sft", "rlvr", "srl", "srl_rlvr"]
        mode_order = ["greedy", "avg1", "avg32"]
        
        pivot = pivot.reindex([m for m in model_order if m in pivot.index])
        pivot = pivot[[m for m in mode_order if m in pivot.columns]]
        
        x = np.arange(len(pivot.index))
        #width = 0.15  # Thinner bars (reduced from 0.25)
        width = 0.02
        
        for i, mode in enumerate(pivot.columns):
            offset = (i - len(pivot.columns) / 2 + 0.5) * width
            # Convert values to percentage for display
            bars = ax.bar(x + offset, pivot[mode].values * 100, width, label=mode.upper())
            
            # Add value labels (convert to percentage)
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           format_percentage(height),
                           ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{benchmark.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([get_model_display_name(m) for m in pivot.index], fontsize=9)
        ax.legend(title='Mode', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.suptitle('Model Performance Across All Modes', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-mode comparison to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison graphs for model evaluation results"
    )
    parser.add_argument(
        "summary_file",
        type=str,
        help="Path to evaluation summary JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Output directory for plots (default: results/plots)"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["greedy", "avg1", "avg32"],
        default=["greedy", "avg1", "avg32"],
        help="Modes to plot (default: all)"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg", "all"],
        default="png",
        help="Output format (default: png)"
    )
    
    args = parser.parse_args()
    
    summary_path = Path(args.summary_file)
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        return
    
    print(f"Loading summary from: {summary_path}")
    df = load_evaluation_summary(summary_path)
    
    if df.empty:
        print("Error: No successful evaluations found in summary.")
        return
    
    print(f"Loaded {len(df)} successful evaluations")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Benchmarks: {sorted(df['benchmark'].unique())}")
    print(f"Modes: {sorted(df['mode'].unique())}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    formats = [args.format] if args.format != "all" else ["png", "pdf"]
    
    # Generate plots for each mode
    for mode in args.modes:
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            print(f"Skipping mode {mode}: no data")
            continue
        
        for fmt in formats:
            # Bar chart
            create_bar_chart(
                df, 
                output_dir / f"bar_chart_{mode}.{fmt}",
                mode=mode
            )
            
            # Line plot
            create_line_plot(
                df,
                output_dir / f"line_plot_{mode}.{fmt}",
                mode=mode
            )
            
            # Heatmap
            create_heatmap(
                df,
                output_dir / f"heatmap_{mode}.{fmt}",
                mode=mode
            )
    
    # Multi-mode comparison
    for fmt in formats:
        create_multi_mode_comparison(
            df,
            output_dir / f"multi_mode_comparison.{fmt}"
        )
    
    # Comprehensive single graph with all models
    for fmt in formats:
        create_comprehensive_bar_chart(
            df,
            output_dir / f"comprehensive_all_models.{fmt}"
        )
    
    # Benchmark(mode) format plot with benchmark at top and models in corner
    for fmt in formats:
        create_benchmark_mode_plot(
            df,
            output_dir / f"benchmark_mode_plot.{fmt}"
        )
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
