#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate orb-tune
python matbenchmark.py --model_type orb --batch_size 96 --normalize_method mean_std
python matbenchmark.py --model_type orb --batch_size 96 --freeze_backbone --normalize_method mean_std

# conda activate jmp-tune
# python matbenchmark.py --model_type jmp --batch_size 4

# conda activate eqv2-tune
# python matbenchmark.py --model_type eqv2 --batch_size 8