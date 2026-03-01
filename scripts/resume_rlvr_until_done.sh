#!/bin/bash
# Resume RLVR GRPO training from the last checkpoint until training finishes.
#
# Usage:
#   bash scripts/resume_rlvr_until_done.sh [TRAIN_ARGS...]
# 
# Examples:
#   # Default output-dir (checkpoints/srl_rlvr), init-from SRL checkpoint
#   bash scripts/resume_rlvr_until_done.sh \
#     --init-from checkpoints/srl/step_500 \
#     --output-dir checkpoints/srl_rlvr
#
#   # Custom dataset / options
#   bash scripts/resume_rlvr_until_done.sh \
#     --init-from checkpoints/srl/step_500 \
#     --output-dir checkpoints/srl_rlvr_qwen7b \
#     --max-train-samples 512
#
# Behavior:
#   - Finds the latest HuggingFace checkpoint directory under --output-dir
#     (e.g. checkpoints/srl_rlvr/checkpoint-50).
#   - If a checkpoint is found, passes --resume-from-checkpoint <latest>
#     to src.train_rlvr_grpo.
#   - If no checkpoint exists, starts training from scratch (no resume flag).
#   - Training then runs to completion in a single invocation.

set -e

# Directory of this script and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Find Python executable - try python3 first, then python, then check conda
if command -v python3 &> /dev/null; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python &> /dev/null; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  # Try to activate conda environment if available
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    # Try to activate a common environment name (adjust if needed)
    if conda env list | grep -q "base"; then
      ENV_NAME=$(conda env list | grep -E "base" | head -n 1 | awk '{print $1}')
      conda activate "$ENV_NAME" 2>/dev/null || true
    fi
    # Now try to find python
    if command -v python3 &> /dev/null; then
      PYTHON_BIN="${PYTHON_BIN:-python3}"
    elif command -v python &> /dev/null; then
      PYTHON_BIN="${PYTHON_BIN:-python}"
    else
      echo "Error: Could not find python or python3. Please ensure Python is installed and in PATH."
      exit 1
    fi
  else
    echo "Error: Could not find python or python3. Please ensure Python is installed and in PATH."
    exit 1
  fi
fi

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version || echo "Warning: Could not get Python version"

# Default output directory (can be overridden by --output-dir in args)
DEFAULT_OUTPUT_DIR="checkpoints/srl_rlvr"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"

# Capture all original arguments
ARGS=("$@")

# Parse --output-dir from ARGS (if present) to know where to look for checkpoints
idx=0
while [ $idx -lt ${#ARGS[@]} ]; do
  case "${ARGS[$idx]}" in
    --output-dir)
      if [ $((idx + 1)) -lt ${#ARGS[@]} ]; then
        OUTPUT_DIR="${ARGS[$((idx + 1))]}"
      fi
      idx=$((idx + 2))
      ;;
    *)
      idx=$((idx + 1))
      ;;
  esac
done

echo "============================================="
echo "RLVR GRPO Resume Script"
echo "Project root : $PROJECT_ROOT"
echo "Output dir   : $OUTPUT_DIR"
echo "Python bin   : $PYTHON_BIN"
echo "Train args   : ${ARGS[*]}"
echo "============================================="
echo ""

# Find latest checkpoint directory under OUTPUT_DIR (HuggingFace Trainer style)
LATEST_CKPT=""
if [ -d "$OUTPUT_DIR" ]; then
  # Look for subdirectories like checkpoint-*, sorted by version
  LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
fi

if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
  echo "Found existing checkpoint: $LATEST_CKPT"
  echo "Resuming RLVR training from last checkpoint..."
else
  echo "No existing checkpoints found under $OUTPUT_DIR."
  echo "Starting RLVR training from scratch (no --resume-from-checkpoint)."
fi

echo ""
echo "Launching training..."
echo "---------------------------------------------"

# Build command
CMD=("$PYTHON_BIN" -m src.train_rlvr_grpo "${ARGS[@]}")
if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
  CMD+=(--resume-from-checkpoint "$LATEST_CKPT")
fi

printf 'Command:'
for tok in "${CMD[@]}"; do
  printf ' %q' "$tok"
done
printf '\n\n'

"${CMD[@]}"
EXIT_CODE=$?

echo "---------------------------------------------"
echo "Training finished with exit code: $EXIT_CODE"
echo "If interrupted again, re-run this script to resume from the latest checkpoint."
echo "============================================="

exit "$EXIT_CODE"

