#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

task_name="matbench_phonons"

model_list=(
    "eqv2"
    "orb"
    "jmp"
)

fold_index=0
train_split=0.9
batch_size=4
max_epochs=500
normalize_method="none"
property_reduction="mean"

for model_type in "${model_list[@]}"; do
    conda activate $model_type-tune

    python matbenchmark-foldx.py \
        --model_type $model_type \
        --fold_index $fold_index \
        --task $task_name \
        --train_split $train_split \
        --batch_size $batch_size \
        --max_epochs $max_epochs \
        --normalize_method $normalize_method \
        --property_reduction $property_reduction \
        --devices 1 2 3 4 5 6 7 \
        --skip_inference
    
    python matbenchmark-foldx.py \
        --model_type $model_type \
        --fold_index $fold_index \
        --task $task_name \
        --train_split $train_split \
        --batch_size $batch_size \
        --max_epochs $max_epochs \
        --normalize_method $normalize_method \
        --property_reduction $property_reduction \
        --devices 1 2 3 4 5 6 7 \
        --skip_tuning
done