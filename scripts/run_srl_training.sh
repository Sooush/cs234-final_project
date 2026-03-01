#!/bin/bash
# Run SRL training only (TRL GRPOTrainer, step-wise reward per paper 2510.25992).
# Optionally prepare data first: python -m src.data_prep --output data/srl_instances.jsonl

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="checkpoints/srl"

echo "SRL training (TRL GRPOTrainer)"
echo "  output-dir: $OUTPUT_DIR"
echo "  init-from:   ${INIT_FROM:-<base model>}"

ARGS=(--output-dir "$OUTPUT_DIR")
[[ -f data/srl_instances.jsonl ]] && ARGS+=(--data-path data/srl_instances.jsonl)
ARGS+=(--resume-latest)

python -m src.train_srl "${ARGS[@]}"
