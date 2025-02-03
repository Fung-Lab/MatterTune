#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh


model_types=(
    # "eqv2"
    # "orb-v2"
    "jmp-s"
    # "jmp-l"
    # "mattersim-1m"
    # "mattersim-5m"
)
# choose from eqv2, orb-v2, jmp-s, jmp-l, mattersim-1m, mattersim-5m
device=2

for model_type in "${model_types[@]}"; do
    python matbench-discovery.py \
        --model_type $model_type \
        --devices $device
done