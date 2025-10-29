#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate schnet-tune

python single-gpu-md_speed_scaling.py --nl_fn_type "pymatgen" --skin_cutoff 1.0 

# python single-gpu-md_speed_scaling.py --nl_fn_type "vesin"

python single-gpu-md_speed_scaling.py --nl_fn_type "vesin" --skin_cutoff 1.0

# python single-gpu-md_speed_scaling.py --nl_fn_type "matscipy"

python single-gpu-md_speed_scaling.py --nl_fn_type "matscipy" --skin_cutoff 1.0