#!/bin/bash
# Run all 11 embodied benchmarks with Molmo2 + vLLM
#
# Usage:
#   ./run_all_benchmarks.sh [MODEL_NAME] [OUTPUT_DIR]
#
# Example:
#   ./run_all_benchmarks.sh Molmo2-4B ./results
#   ./run_all_benchmarks.sh Molmo2-8B ./results

set -e

# Default values
MODEL_NAME=${1:-"Molmo2-4B"}
OUTPUT_DIR=${2:-"./outputs"}

# vLLM environment settings
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

# Conda environment (uncomment if needed)
# source activate mm_olmo

echo "=============================================="
echo "Running all 11 embodied benchmarks"
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# List of all datasets to evaluate (12 total from 11 benchmarks)
# Note: RoboSpatial has 2 tasks (Pointing and VQA)
# Note: Skipped benchmarks: MinimalVideos (too large), OpenEQA (needs manual download)

DATASETS=(
    # Image MCQ Benchmarks
    "CVBench_Embodied"
    "EmbSpatial_Embodied"
    "SAT_Embodied"
    "BLINK_Embodied"
    "MindCube_Embodied"

    # Image Pointing Benchmarks
    "RefSpatial_Embodied"
    "RoboSpatial_Pointing"
    "RoboSpatial_VQA"
    "Where2Place_Embodied"

    # Video Benchmarks
    "VSI_Bench_Embodied"
    "CosmosReason1_Embodied"

    # Special Format
    "ERQA_Embodied"
)

# Run each benchmark
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running: ${dataset}"
    echo "=============================================="

    python run.py \
        --model "${MODEL_NAME}" \
        --data "${dataset}" \
        --work-dir "${OUTPUT_DIR}" \
        --use-vllm \
        --verbose

    echo "Completed: ${dataset}"
done

echo ""
echo "=============================================="
echo "All benchmarks completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
