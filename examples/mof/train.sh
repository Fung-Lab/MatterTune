#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh


# conda activate mattersim-tune
# batch_size=24
# python mof_train.py \
#     --batch_size $batch_size \
#     --model_type "mattersim-1m" \
#     --devices 0 1 2 3 4 5 6 7 \
#     --conservative

# python mof_train.py \
#     --batch_size $batch_size \
#     --model_type "mattersim-5m" \
#     --devices 0 1 2 3 4 5 6 7 \
#     --conservative

# conda activate mace-tune
# batch_size=24
# python mof_train.py \
#     --batch_size $batch_size \
#     --model_type "mace-medium-omat-0" \
#     --devices 0 1 2 3 4 5 6 7 \
#     --conservative

# conda activate jmp-tune
# batch_size=2
# python mof_train.py \
#     --batch_size $batch_size \
#     --model_type "jmp-s" \
#     --devices 0 1 2 3 4 5 6 7 \
#     --conservative

conda activate uma-tune
batch_size=8
python mof_train.py \
    --batch_size $batch_size \
    --model_type "uma-s-1.1" \
    --devices 0 1 2 3 4 5 6 7 \
    --conservative
batch_size=4
python mof_train.py \
    --batch_size $batch_size \
    --model_type "uma-m-1.1" \
    --devices 0 1 2 3 4 5 6 7 \
    --conservative