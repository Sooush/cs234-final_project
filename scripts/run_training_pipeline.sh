#!/bin/bash
# End-to-end training pipeline: SFT → SRL → RLVR
# Uses paper-style hyperparameters via the Python scripts in src/.
#
# Stages:
#   1) SFT:  python -m src.train_sft        → checkpoints/sft
#   2) SRL:  python -m src.train_srl        → checkpoints/srl
#   3) RLVR: python -m src.train_rlvr_grpo  → checkpoints/srl_rlvr
#
# After this, you can run evaluation (e.g. scripts/run_math_evaluations.sh).

set -e
cd "$(dirname "$0")/.."

SFT_OUTPUT="checkpoints/sft"
SRL_OUTPUT="checkpoints/srl"
SRL_INIT_FROM="checkpoints/sft_merged"
RLVR_OUTPUT="checkpoints/srl_rlvr"
RLVR_INIT_FROM="checkpoints/srl_merged"

DEVICE_ARG=""

usage() {
  echo "Usage: $0 [--skip-sft] [--skip-srl] [--skip-rlvr] [--device DEVICE]"
  echo ""
  echo "Examples:"
  echo "  $0                       # Run all stages SFT → SRL → RLVR"
  echo "  $0 --skip-sft            # Assume SFT already done; run SRL → RLVR"
  echo "  $0 --device mps          # Use MPS where applicable"
  exit 1
}

SKIP_SFT=0
SKIP_SRL=0
SKIP_RLVR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sft)  SKIP_SFT=1; shift ;;
    --skip-srl)  SKIP_SRL=1; shift ;;
    --skip-rlvr) SKIP_RLVR=1; shift ;;
    --device)    DEVICE_ARG="--device $2"; shift 2 ;;
    --help|-h)   usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

echo "========================================"
echo "Training Pipeline: SFT → SRL → RLVR"
echo "Base model: configured in configs/models_config.json (base_model)"
echo "Outputs:"
echo "  SFT   → $SFT_OUTPUT"
echo "  SRL   → $SRL_OUTPUT"
echo "  RLVR  → $RLVR_OUTPUT"
echo "========================================"
echo ""

# 1) SFT
if [[ $SKIP_SFT -eq 0 ]]; then
  echo "[Stage 1/3] SFT training..."
  python -m src.train_sft \
    --output-dir "$SFT_OUTPUT"
  echo ""
else
  echo "[Stage 1/3] SFT training skipped (--skip-sft)"
fi

# 2) SRL (init-from SFT by default)
if [[ $SKIP_SRL -eq 0 ]]; then
  echo "[Stage 2/3] SRL training (initialized from $SRL_INIT_FROM)..."
  python -m src.train_srl \
    --init-from "$SRL_INIT_FROM" \
    --output-dir "$SRL_OUTPUT" \
    --data data/srl_instances.jsonl \
    --num-steps 500 \
    --batch-size 4 \
    --group-size 8 \
    --max-new-tokens 512 \
    --temperature 1.0 \
    --lr 5e-7 \
    --clip-epsilon 0.2 \
    --eps-std 0.01 \
    --checkpoint-every 100 \
    --config configs/srl_qwen7b.yaml
  echo ""
else
  echo "[Stage 2/3] SRL training skipped (--skip-srl)"
fi

# 3) RLVR (init-from SRL checkpoint)
if [[ $SKIP_RLVR -eq 0 ]]; then
  LATEST_CKPT=""
  if [ -d "$RLVR_OUTPUT" ]; then
    # Look for subdirectories like checkpoint-*, sorted by version
    LATEST_CKPT=$(ls -d "$RLVR_OUTPUT"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
  fi

  if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
    echo "Found existing checkpoint: $LATEST_CKPT"
    echo "Resuming RLVR training from last checkpoint..."
  else
    echo "No existing checkpoints found under $RLVR_OUTPUT."
    echo "Starting RLVR training from scratch (no --resume-from-checkpoint)."
  fi

  RLVR_ARGS=(
    --max-train-samples 250
    --init-from "$RLVR_INIT_FROM"
    --output-dir "$RLVR_OUTPUT"
  )
  
  if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
    RLVR_ARGS+=(--resume-from-checkpoint "$LATEST_CKPT")
  fi

  echo "[Stage 3/3] RLVR GRPO training (initialized from $RLVR_INIT_FROM)..."
  python -m src.train_rlvr_grpo "${RLVR_ARGS[@]}"
  echo ""
else
  echo "[Stage 3/3] RLVR training skipped (--skip-rlvr)"
fi

echo "========================================"
echo "Training pipeline complete."
echo "SFT   checkpoint: $SFT_OUTPUT"
echo "SRL   checkpoint: $SRL_OUTPUT"
echo "RLVR  checkpoint: $RLVR_OUTPUT"
echo "========================================"

