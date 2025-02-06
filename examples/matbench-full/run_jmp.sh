#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

model_type="jmp"
fold_index=0
train_split=0.9
max_epochs=500

conda activate $model_type-tune

python matbenchmark-foldx.py \
    --model_type $model_type \
    --fold_index $fold_index \
    --task matbench_phonons \
    --train_split $train_split \
    --batch_size 16 \
    --max_epochs $max_epochs \
    --normalize_method reference \
    --property_reduction mean \
    --devices 0 1 2 3 4 5 6 7 \
    --load_best_ckpt

python matbenchmark-foldx.py \
    --model_type $model_type \
    --fold_index $fold_index \
    --task matbench_perovskites \
    --train_split $train_split \
    --batch_size 16 \
    --max_epochs $max_epochs \
    --normalize_method none \
    --property_reduction mean \
    --devices 0 1 2 3 4 5 6 7 \
    --load_best_ckpt

python matbenchmark-foldx.py \
    --model_type $model_type \
    --fold_index $fold_index \
    --task matbench_mp_e_form \
    --train_split $train_split \
    --batch_size 4 \
    --max_epochs $max_epochs \
    --normalize_method reference \
    --property_reduction mean \
    --devices 0 1 2 3 4 5 6 7 \
    --load_best_ckpt

python matbenchmark-foldx.py \
    --model_type $model_type \
    --fold_index $fold_index \
    --task matbench_mp_gap \
    --train_split $train_split \
    --batch_size 4 \
    --max_epochs $max_epochs \
    --normalize_method none \
    --property_reduction mean \
    --devices 0 1 2 3 4 5 6 7 \
    --load_best_ckpt