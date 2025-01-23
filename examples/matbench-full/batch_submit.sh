#!/bin/bash

# 定义任务列表，格式为：task_name
list_of_tasks=(
    "matbench_dielectric"
    "matbench_expt_gap"
    "matbench_jdft2d"
    "matbench_log_gvrh"
    "matbench_log_kvrh"
    "matbench_mp_e_form"
    "matbench_mp_gap"
    "matbench_perovskites"
    "matbench_phonons"
    "matbench_steels"
)

# 共享的超参数
model_type="jmp"
fold_index=0
train_split=0.9
batch_size=4
max_epochs=500
devices="0,1,2,3"
property_reduction="mean"

# 循环生成和提交任务
for task_name in "${list_of_tasks[@]}"; do
    # 生成唯一的作业脚本文件名
    job_file="batchjob_${task_name}_fold${fold_index}.sh"

    # 创建 Slurm 作业脚本
    cat <<EOF > $job_file
#!/bin/bash
#SBATCH --job-name=${model_type}-${task_name}-Fold${fold_index}      # 作业名称
#SBATCH --nodes=1                     # 使用 1 个节点
#SBATCH --ntasks-per-node=4           # 每个节点运行 4 个任务
#SBATCH --gpus-per-task=1             # 每个任务分配 1 张 GPU
#SBATCH --cpus-per-task=4             # 每个任务使用 4 个 CPU 核心
#SBATCH --qos=regular                 # QOS 类型
#SBATCH --time=48:00:00               # 最长运行时间
#SBATCH -C gpu                        # 使用 GPU 节点
#SBATCH -A m4555_g                    # 项目账户名

module load python  # 加载 Python 模块
conda activate jmp-tune  # 激活环境

# 使用 Slurm 自动分配任务
srun python -u matbenchmark-foldx.py \\
    --model_type "$model_type" \\
    --fold_index $fold_index \\
    --task $task_name \\
    --train_split $train_split \\
    --batch_size $batch_size \\
    --max_epochs $max_epochs \\
    --devices $devices \\
    --property_reduction "$property_reduction"
EOF

    # 提交作业
    sbatch $job_file

    # 提示信息
    echo "Submitted job: $job_file"
done
