#!/bin/bash
# Run Molmo2-4B (original) evaluation on all embodied benchmarks (except VSI_Bench)

# Activate conda environment
source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/etc/profile.d/conda.sh
conda activate mm_olmo

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0

# Run evaluation
python run_full_eval.py \
    --model Molmo2-4B \
    --max_new_tokens 512 \
    --output results \
    --benchmarks \
        CVBench_Embodied \
        EmbSpatial_Embodied \
        SAT_Embodied \
        BLINK_Embodied \
        RefSpatial_Embodied \
        RoboSpatial_Pointing \
        RoboSpatial_VQA \
        MindCube_Tiny_Embodied \
        Where2Place_Embodied \
        CosmosReason1_Embodied \
        MMSI_Bench_Embodied
