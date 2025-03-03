#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

models=(
    # "mattersim-1m"
    # "orb-v2"
    # "jmp-s"
    "eqv2"
)

batch_size=16

for model in "${models[@]}"; do
    if [ $model == "mattersim-1m" ]; then
        conda activate mattersim-tune
        batch_size=16
    elif [ $model == "orb-v2" ]; then
        conda activate orb-tune
        batch_size=8
    elif [ $model == "jmp-s" ]; then
        conda activate jmp-tune
        batch_size=2
    elif [ $model == "eqv2" ]; then
        conda activate eqv2-tune
        batch_size=4
    fi
    python water-finetune.py \
        --model_type $model \
        --train_down_sample 900 \
        --batch_size $batch_size
    python water-finetune.py \
        --model_type $model \
        --train_down_sample 30 \
        --batch_size $batch_size
    python water-finetune.py --down_sample_refill \
        --model_type $model \
        --train_down_sample 30  \
        --batch_size $batch_size
done