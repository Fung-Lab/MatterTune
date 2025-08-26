#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh


# conda activate mattersim-tune
# python Li3PO4_speedtest.py \
#     --ckpt_path "./checkpoints/mattersim-1m-best-MPx1-first_half_0.02.ckpt" \
#     --expected_size 421824 \
#     --devices 0 \
#     --granularity 4 
# python Li3PO4_speedtest.py \
#     --ckpt_path "./checkpoints/mattersim-1m-best-MPx1-first_half_0.02.ckpt" \
#     --expected_size 1000000 \
#     --devices 0 \
#     --granularity 5 
# python Li3PO4_speedtest.py \
#     --ckpt_path "./checkpoints/mattersim-1m-best-MPx1-first_half_0.02.ckpt" \
#     --expected_size 5000000 \
#     --devices 0 \
#     --granularity 8 
# python Li3PO4_speedtest.py \
#     --ckpt_path "./checkpoints/mattersim-1m-best-MPx1-first_half_0.02.ckpt" \
#     --expected_size 10000000 \
#     --devices 0 \
#     --granularity  10


# conda activate mace-tune
# python Li3PO4_speedtest.py \
#     --ckpt_path "./checkpoints/mace-medium-omat-0-best-MPx1-first_half_0.02.ckpt" \
#     --expected_size 421824 \
#     --devices 0 \
#     --granularity 4 

# python Li3PO4_speedtest.py \
#     --ckpt_path "./checkpoints/mace-medium-omat-0-best-MPx1-first_half_0.02.ckpt" \
#     --expected_size 1000000 \
#     --devices 0 \
#     --granularity 6 

# python Li3PO4_speedtest.py \
#     --ckpt_path "./checkpoints/mace-medium-omat-0-best-MPx1-first_half_0.02.ckpt" \
#     --expected_size 5000000 \
#     --devices 0 \
#     --granularity 10 


conda activate orbv3-tune
# python Li3PO4_speedtest.py \
#     --ckpt_path "./checkpoints/orb-v3-conservative-inf-omat-best-MPx1-first_half_0.02.ckpt" \
#     --expected_size 421824 \
#     --devices 0 \
#     --granularity 4 

# python Li3PO4_speedtest.py \
#     --ckpt_path "./checkpoints/orb-v3-conservative-inf-omat-best-MPx1-first_half_0.02.ckpt" \
#     --expected_size 1000000 \
#     --devices 0 \
#     --granularity 6 

python Li3PO4_speedtest.py \
    --ckpt_path "./checkpoints/orb-v3-conservative-inf-omat-best-MPx1-first_half_0.02.ckpt" \
    --expected_size 5000000 \
    --devices 0 \
    --granularity 10 