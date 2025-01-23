#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

# conda activate orb-tune
# python matbenchmark.py --model_type orb --batch_size 96 --normalize_method reference --fold_index 0

conda activate jmp-tune
python matbenchmark-foldx.py --model_type jmp --batch_size 4 --normalize_method none --devices 0 1 2 3 4 5 6 7

# conda activate eqv2-tune
# python matbenchmark.py --model_type eqv2 --batch_size 8


# # Plot Y-Distribution
# conda activate jmp-tune
# python y_distribution.py --task matbench_mp_gap --normalize_method reference
# python y_distribution.py --task matbench_mp_gap --normalize_method mean_std
# python y_distribution.py --task matbench_mp_gap --normalize_method rms

# python y_distribution.py --task matbench_log_kvrh --normalize_method reference
# python y_distribution.py --task matbench_log_kvrh --normalize_method mean_std
# python y_distribution.py --task matbench_log_kvrh --normalize_method rms

# python y_distribution.py --task matbench_perovskites --normalize_method reference
# python y_distribution.py --task matbench_perovskites --normalize_method mean_std
# python y_distribution.py --task matbench_perovskites --normalize_method rms