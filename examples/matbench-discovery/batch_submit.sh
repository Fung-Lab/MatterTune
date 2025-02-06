#!/bin/bash

list_of_models=(
    "eqv2",
    "orb",
    "mattersim",
)

save_dir="./results"

for model in "${list_of_models[@]}"; do
    l_idx=0
    while [ $l_idx -lt 260000 ]; do
        r_idx=$((l_idx + 2500))
        job_file="mb-discovery-${model}-${l_idx}-${r_idx}.sh"
        cat <<EOF > $job_file
#!/bin/bash
#SBATCH --job-name=mb-discovery-${model}-${l_idx}-${r_idx}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH -C gpu&hbm40g
#SBATCH -A m4555_g

module load python
conda activate ${model}-mbd

# 使用 Slurm 自动分配任务
srun python split_relax.py \\
    --model_type $model \\
    --l_idx $l_idx \\
    --r_idx $r_idx \\
    --device 0 \\
    --save_dir $save_dir \\
EOF

        sbatch $job_file
        echo "Submitted job: $job_file"
        l_idx=$((l_idx + 2500))
    done
done