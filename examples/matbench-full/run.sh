#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

list_of_tasks=(
    # "matbench_dielectric"
    "matbench_jdft2d"
    "matbench_log_gvrh"
    "matbench_log_kvrh"
    "matbench_perovskites"
    "matbench_phonons"
    "matbench_mp_e_form"
    "matbench_mp_gap"
)

model_type="jmp"
fold_index=0
train_split=0.9
batch_size=4
max_epochs=500
normalize_method="none"
property_reduction="mean"

conda activate $model_type-tune

for task_name in "${list_of_tasks[@]}"; do
    python matbenchmark-foldx.py \
        --model_type $model_type \
        --fold_index $fold_index \
        --task $task_name \
        --train_split $train_split \
        --batch_size $batch_size \
        --max_epochs $max_epochs \
        --normalize_method $normalize_method \
        --property_reduction $property_reduction \
        --devices 0 1 2 3 4 5 6 7 \
        --load_best_ckpt
done