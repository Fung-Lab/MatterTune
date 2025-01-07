#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate jmp-tune
python jmp-finetune.py --conservative True --down_sample_refill True
python jmp-finetune.py --conservative False --down_sample_refill True