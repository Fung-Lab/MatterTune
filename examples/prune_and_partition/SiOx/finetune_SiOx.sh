#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh


conda activate mattersim-tune
batch_size=16
python train.py \
    --model_type "mattersim-1m" \
    --batch_size $batch_size \
    --early_stop_mp_steps 2 \
    --down_sample 0.02 \
    --devices 0 1 2 3 4 5 6 7