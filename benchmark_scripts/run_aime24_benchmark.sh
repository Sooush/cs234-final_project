#!/bin/bash

# This prevents the "Invalid buffer size" error on macOS MPS
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Optional: Prevents some memory fragmentation issues on Mac
export PYTORCH_ENABLE_MPS_FALLBACK=1

# --- Configuration ---
LOG_TIME=$(date +%s)

ATTEMPTS=1

OUTPUT_DIR="./results_aime24_avg$ATTEMPTS"
OUTPUT_LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_LOG_DIR"

LOG_FILE="$OUTPUT_LOG_DIR/benchmark_aime24_$LOG_TIME.log"

# ----- Pre-check check if the model is loaded into the GPU's memory space
echo "===========================================" | tee -a "$LOG_FILE"
echo "PRE-CHECK: Hardware & Software Audit" | tee -a "$LOG_FILE"

# --- 2. AUTOMATED DEVICE CHECK ---
# Checks if MPS is available and if the model can be loaded to it
python3 << END
import torch
import sys
mps_avail = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()
print(f"PyTorch Version: {torch.__version__}")
print(f"MPS Available:   {mps_avail}")
print(f"MPS Built:       {mps_built}")

if not mps_avail:
    print("ERROR: MPS GPU not detected. Aborting to save your CPU!")
    sys.exit(1)
print("GPU CHECK PASSED: M4 Metal Backend Ready.")
END


# --- Header ---
echo "===========================================" | tee -a "$LOG_FILE"
echo "Starting AIME24 Benchmark: Qwen 2.5-7B" | tee -a "$LOG_FILE"
echo "Start Time: $(date)" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"

# Record start time in seconds
START_TIME=$(date +%s)

# --- Execute Command ---
# We use 'tee' to show output in terminal AND save to log file
# Note: The 'time' command output is redirected to the log
{
  time lm_eval --model local-completions \
      --model_args model=Qwen/Qwen2.5-7B-Instruct,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
      --tasks aime24 \
      --device mps \
      --batch_size 1 \
      --gen_kwargs temperature=1.0,do_sample=True,n=$ATTEMPTS,max_gen_toks=4096 \
      --output_path "$OUTPUT_DIR" \
      --log_samples
} 2>&1 | tee -a "$LOG_FILE"

# --- Footer & Timing ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Convert seconds to hours, minutes, seconds
HOURS=$((DURATION / 3600))
MINS=$(( (DURATION % 3600) / 60 ))
SECS=$(( DURATION % 60 ))

echo "" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
echo "Benchmark Finished!" | tee -a "$LOG_FILE"
echo "End Time:   $(date)" | tee -a "$LOG_FILE"
printf "Total Duration: %02d:%02d:%02d (HH:MM:SS)\n" $HOURS $MINS $SECS | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
