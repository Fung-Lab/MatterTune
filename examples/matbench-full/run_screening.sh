#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

# conda activate orb-tune
# python screening.py --ckpt_path ./checkpoints-matbench_mp_gap/orb-best-fold0.ckpt

# conda activate jmp-tune
# python screening.py --ckpt_path ./checkpoints-matbench_mp_gap/jmp-best-fold0.ckpt

conda activate eqv2-tune
python screening.py --ckpt_path ./checkpoints-matbench_mp_gap/eqv2-best-fold0.ckpt