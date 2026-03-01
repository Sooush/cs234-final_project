#!/bin/bash
# Evaluate all models on all mathematical reasoning benchmarks
# Generates evaluation results and comparison graphs

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "Mathematical Reasoning Benchmarks Evaluation"
echo "========================================"
echo "Benchmarks: AMC23, AIME24, AIME25, Minerva Math"
echo "Models: Base, SFT, SRL, SRL→RLVR"
echo "Modes: greedy, avg1, avg32"
echo "========================================"
echo ""

# Step 1: Run all evaluations
echo "Step 1: Running evaluations..."
#python -m src.eval_all_benchmarks \
#  --models base sft srl srl_rlvr \
#  --benchmarks amc23 aime24 aime25 minerva_math \
#  --modes greedy avg1 avg32 \
#  --max-gen-toks 4096 \
#  --config configs/models_config.json


python -m src.eval_all_benchmarks \
  --models base \
  --benchmarks minerva_math \
  --modes greedy avg1 \
  --max-gen-toks 4096 \
  --results-dir results-0228-0216 \
  --checkpoint-file results-0228-0216/eval_checkpoint.json \
  --cache-dir /mnt/persistent/lm_eval_cache \
  --config configs/models_config.json


# test run
#python -m src.eval_all_benchmarks \
#  --models base \
#  --benchmarks aime24 aime25 \
#  --modes greedy \
#  --max-gen-toks 4096 \
#  --config configs/models_config.json \
#  --limit 1 \
#  --device mps

# Step 2: Find the most recent summary file
SUMMARY_FILE=$(ls -t results/evaluation_summary_*.json 2>/dev/null | head -1)

if [ -z "$SUMMARY_FILE" ]; then
    echo "Error: No evaluation summary file found"
    exit 1
fi

echo ""
echo "Step 2: Aggregating results..."
python -m src.aggregate_results "$SUMMARY_FILE" --format all

echo ""
echo "Step 3: Generating comparison graphs..."
python -m src.plot_comparison "$SUMMARY_FILE" --format all

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results summary: $SUMMARY_FILE"
echo "Plots saved to: results/plots/"
echo "========================================"
