#!/bin/bash
#SBATCH --job-name=4GPU_training      # 作业名称
#SBATCH --nodes=1                     # 使用 1 个节点
#SBATCH --ntasks-per-node=4           # 每个节点运行 4 个任务
#SBATCH --gpus-per-task=1             # 每个任务分配 1 张 GPU
#SBATCH --cpus-per-task=4             # 每个任务使用 4 个 CPU 核心
#SBATCH --qos=regular                 # QOS 类型
#SBATCH --time=48:00:00               # 最长运行时间
#SBATCH -C gpu                        # 使用 GPU 节点
#SBATCH -A m3641_g                    # 项目账户名

module load python  # 加载 Python 模块
conda activate orb-tune  # 激活环境

cd ./MatterTune/examples/matbench-full

# 使用 Slurm 自动分配任务
srun python -u matbenchmark.py \
    --model_type "orb" \
    --fold_index 0 \
    --task matbench_mp_gap \
    --train_split 0.9 \
    --batch_size 96 \
    --max_epochs 1000 \
    --devices 0 1 2 3
