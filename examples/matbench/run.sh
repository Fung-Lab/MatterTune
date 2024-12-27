#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

# Activate the environment and run the script
conda activate jmp-tune
python jmp-finetune.py --task matbench_log_kvrh
python jmp-finetune.py --task matbench_perovskites

conda activate orb-tune
python orb-finetune.py --task matbench_log_kvrh
python orb-finetune.py --task matbench_perovskites

conda activate eqv2-tune
python eqv2-finetune.py --task matbench_log_kvrh
python eqv2-finetune.py --task matbench_perovskites