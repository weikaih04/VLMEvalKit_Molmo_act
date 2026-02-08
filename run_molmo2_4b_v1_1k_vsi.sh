#!/bin/bash
# Run Molmo2-4B-spatial-tuning-v1-1k (finetuned) on VSI-Bench

source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/etc/profile.d/conda.sh
conda activate mm_olmo

export CUDA_VISIBLE_DEVICES=0

python run_full_eval.py \
    --model molmo2-4-spatial-tuning-v1-1k \
    --max_new_tokens 512 \
    --output results \
    --benchmarks vsibench_16frame
