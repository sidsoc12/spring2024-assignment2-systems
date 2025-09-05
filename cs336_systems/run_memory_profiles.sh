#!/bin/bash
set -e

echo "--- Starting Memory Profiling Sweep ---"

# --- FP32 Analysis ---
echo ""
echo "--- Profiling FP32 ---"
# Inference Only
echo "Running: FP32, Context 128, Inference"
uv run python memory_profiling.py --context_length 128
echo "Running: FP32, Context 256, Inference"
uv run python memory_profiling.py --context_length 256
echo "Running: FP32, Context 512, Inference"
uv run python memory_profiling.py --context_length 512

# Full Training Step
echo "Running: FP32, Context 128, Full Train Step"
uv run python memory_profiling.py --context_length 128 --full_train_step
echo "Running: FP32, Context 256, Full Train Step"
uv run python memory_profiling.py --context_length 256 --full_train_step
echo "Running: FP32, Context 512, Full Train Step"
uv run python memory_profiling.py --context_length 512 --full_train_step

# --- Mixed-Precision (BF16) Analysis ---
echo ""
echo "--- Profiling BF16 ---"
# Inference Only
echo "Running: BF16, Context 512, Inference"
uv run python memory_profiling.py --context_length 512 --precision bf16

# Full Training Step
echo "Running: BF16, Context 512, Full Train Step"
uv run python memory_profiling.py --context_length 512 --precision bf16 --full_train_step

echo ""
echo "--- ✅ Memory Profiling Sweep Complete! ---"
echo "All .pickle files have been generated."