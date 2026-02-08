#!/bin/bash
# Run full evaluation on all embodied benchmarks
#
# Usage:
#   ./run_eval.sh                    # Run all benchmarks
#   ./run_eval.sh --model Molmo2-8B  # Use different model
#   ./run_eval.sh --benchmarks CVBench_Embodied EmbSpatial_Embodied  # Specific benchmarks

# Activate conda environment
source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/etc/profile.d/conda.sh
conda activate mm_olmo

# Set CUDA devices
# Note: Molmo2 with mm_olmo plugin currently has issues with tensor parallelism
# Using single GPU for now, but continuous batching still provides speedup
export CUDA_VISIBLE_DEVICES=0

# Run evaluation
python run_full_eval.py \
    --model Molmo2-4B \
    --max_new_tokens 512 \
    --output results \
    "$@"
