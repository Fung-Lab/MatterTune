#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

export PYTORCH_DONATED_BUFFERS=0
conda activate orbv3-tune
    batch_size=4
    python water-finetune.py \
        --model_type "orb-v3-conservative-inf-omat" \
        --batch_size $batch_size \
        --lr 1e-4 \
        --down_sample_refill \
        --devices 0 1 2 3 4 5 \
        --conservative 