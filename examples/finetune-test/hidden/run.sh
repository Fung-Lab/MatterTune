#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh


# conda activate mattersim-tune
# python md_speed_test.py \
#     --ckpt_path "./checkpoints/mattersim-1m-best-conservative.ckpt"

# conda activate nequip-tune
# python md_speed_test.py \
#     --ckpt_path "./checkpoints/NequIP-OAM-L-0.1-best-30-refill-conservative.ckpt"

# conda activate nequip-tune
# python md_speed_test.py \
#     --ckpt_path "./checkpoints/mir-group__NequIP-OAM-L__0.1.nequip.pth" \
#     --compiled

# conda activate nequip-tune
# python md_speed_test.py \
#     --ckpt_path "./checkpoints/compiled_model_from_mt.nequip.pth" \
#     --compiled

conda activate nequip-tune
python md_speed_test.py \
    --ckpt_path "./checkpoints/mir-group__NequIP-OAM-L__0.1-aotinductor.nequip.pt2" \
    --compiled

conda activate nequip-tune
python md_speed_test.py \
    --ckpt_path "./checkpoints/compiled_model_from_mt-aotinductor.nequip.pt2" \
    --compiled

# conda activate mace-tune
# python md_speed_test.py \
#     --ckpt_path "./checkpoints/mace-medium-best-900-refill-conservative.ckpt"

# conda activate orbv3-tune
# python md_speed_test.py \
#     --ckpt_path "./checkpoints/orb-v3-conservative-inf-omat-best-30.ckpt"

