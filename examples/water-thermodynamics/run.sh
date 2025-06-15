#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate mace-tune
batch_size=12
python water-finetune.py \
    --model_type "mace-medium" \
    --batch_size $batch_size \
    --lr 1e-4 \
    --devices 0 1 2 3 \
    --conservative \
    --train_down_sample 30 \
    --down_sample_refill

python water-finetune.py \
    --model_type "mace-medium" \
    --batch_size $batch_size \
    --lr 1e-4 \
    --devices 0 1 2 3 \
    --conservative \
    --train_down_sample 900 