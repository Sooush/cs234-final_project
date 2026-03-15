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

DATE_TAG=${DATE_TAG:-$(date +%m%d)}

MODELS_DEFAULT="srl"
BENCHMARKS_DEFAULT="aime24 aime25"
MODES_DEFAULT="greedy avg1"

MODELS=""

usage() {
  echo "Usage: $0 [--models MODEL1 MODEL2 ...]"
  echo ""
  echo "Examples:"
  echo "  $0                         # uses default MODELS='${MODELS_DEFAULT}'"
  echo "  $0 --models srl rlvr       # run for SRL and RLVR models only"
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --models)
      shift
      MODELS=""
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        MODELS="${MODELS:+$MODELS }$1"
        shift
      done
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

# Allow overriding via environment variables if CLI not provided.
MODELS=${MODELS:-${MODELS_ENV:-$MODELS_DEFAULT}}
BENCHMARKS=${BENCHMARKS:-$BENCHMARKS_DEFAULT}
MODES=${MODES:-$MODES_DEFAULT}

for model in $MODELS; do
  for bench in $BENCHMARKS; do
    for mode in $MODES; do
      echo "Start run benchmark with model=${model}, benchmark=${bench}, mode=${mode}"
      RESULTS_DIR="results-${DATE_TAG}-${model}-${bench}_${mode}"
      CACHE_DIR="benchmark_cache_dir/persistent/lm_eval_cache_${bench}_${mode}"

      python -m src.eval_all_benchmarks \
        --models "${model}" \
        --benchmarks "${bench}" \
        --modes "${mode}" \
        --max-gen-toks 8192 \
        --results-dir "${RESULTS_DIR}" \
        --checkpoint-file "${RESULTS_DIR}/eval_checkpoint.json" \
        --cache-dir "${CACHE_DIR}" \
        --config configs/models_config.json
    done
  done
done


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