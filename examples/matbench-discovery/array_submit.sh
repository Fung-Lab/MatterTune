#!/bin/bash
#SBATCH --job-name=mb-discovery-array
#SBATCH --array=0-1303
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=50
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH -C gpu&hbm40g
#SBATCH -A m4555_g

module load python
# 按实际需求加载 Python/Conda 模块
# module load python/3.9  (示例)

cd /global/homes/l/lingyu/MatterTune/examples/matbench-discovery

# ---------------------------
# 1. 定义要处理的数据规模
max_lidx=260000

# 2. 为每个模型指定 step_size 和所需分块数
#    (你可以根据需要灵活修改这些数字)
eqv2_step=500
eqv2_chunks=521    # ceil(260000 / 500) = 521

orb_step=500
orb_chunks=521      # ceil(260000 / 500) = 521

mattersim_step=1000
mattersim_chunks=261  # ceil(260000 / 1000) = 261

# 3. 计算总的 array size = 521 + 521 + 261 = 1303

task_id=$SLURM_ARRAY_TASK_ID

# 4. 判定当前 task_id 对应哪个模型
if [ $task_id -lt $eqv2_chunks ]; then
    model="eqv2"
    chunk_id=$task_id
    step_size=$eqv2_step
    n_jobs=4
elif [ $task_id -lt $((eqv2_chunks + orb_chunks)) ]; then
    model="orb"
    chunk_id=$((task_id - eqv2_chunks))
    step_size=$orb_step
    n_jobs=4
else
    model="mattersim"
    chunk_id=$((task_id - eqv2_chunks - orb_chunks))
    step_size=$mattersim_step
    n_jobs=25
fi

# 5. 计算本任务对应的 l_idx, r_idx
l_idx=$((chunk_id * step_size))
r_idx=$((l_idx + step_size))
if [ $r_idx -gt $max_lidx ]; then
    r_idx=$max_lidx
fi

# 6. 打印信息，便于排查
echo "task_id=$task_id, model=$model, chunk_id=$chunk_id, l_idx=$l_idx, r_idx=$r_idx, step_size=$step_size, n_jobs=$n_jobs"

# 7. 激活与 model 对应的 conda 环境（示例：${model}-mbd）
conda activate ${model}-mbd

# 8. 运行主脚本
srun python split_relax.py \
    --model_type $model \
    --l_idx $l_idx \
    --r_idx $r_idx \
    --device 0 \
    --n_jobs $n_jobs \
    --save_dir ./results
