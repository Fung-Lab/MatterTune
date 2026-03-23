#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate uma-elec

python md.py \
    --model-type "uma" \
    --model-name "uma-s-1p1" \
    --task-name "omat" \
    --device "cuda" \
    --structure "./data/LiH2O.xyz" \
    --target-indices "0" \
    --epsilon 1.0 \
    --sigma 1.0 \
    --alpha 0.5 \
    --rc 3.0 \
    --ro 1.5 \
    --temperature 300.0 \
    --timestep-fs 1.0 \
    --friction-fs-inv 0.02 \
    --steps 1000 \
    --log-interval 10 \
    --seed 7 \
    --output-dir "./outputs/md" \
    --trajectory-name "md.xyz" \
    --final-structure-name "final.xyz"