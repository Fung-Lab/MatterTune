#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate mattersim-tune
batch_size=32
python train.py \
    --model_type "mattersim-1m" \
    --batch_size $batch_size \
    --pruned_mp_steps 1 \
    --lr 1e-4 \
    --down_sample "first_half_0.02" \
    --devices 0 1 2 3 4 5 6 7

conda activate jmp-tune
batch_size=8
python train.py \
    --model_type "jmp-s" \
    --batch_size $batch_size \
    --pruned_mp_steps 1 \
    --lr 1e-4 \
    --down_sample "first_half_0.02" \
    --devices 0 1 2 3 4 5 6 7

conda activate orbv3-tune
batch_size=32
python train.py \
    --model_type "orb-v3-conservative-inf-omat" \
    --batch_size $batch_size \
    --pruned_mp_steps 1 \
    --lr 1e-4 \
    --down_sample "first_half_0.02" \
    --devices 0 1 2 3 4 5 6 7 \
    --early_stop_patience 200


conda activate mace-tune
batch_size=16
python train.py \
    --model_type "mace-medium-omat-0" \
    --batch_size $batch_size \
    --pruned_mp_steps 1 \
    --lr 1e-4 \
    --devices 0 1 2 3 4 5 6 7 \
    --conservative 