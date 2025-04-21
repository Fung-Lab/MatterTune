#!/usr/bin/env python3
import os
import time
import pickle
import argparse
import logging

# 如果你愿意，可以用 watchdog 监控目录
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

from ase import Atoms
import torch
import numpy as np

# 下面几行根据你现有代码改写
from mattertune.backbones import (
    JMPBackboneModule,
    MatterSimM3GNetBackboneModule,
    ORBBackboneModule,
    EqV2BackboneModule,
)
from mattertune.wrappers.ase_calculator import grid_partition_atoms
from mattertune.callbacks.multi_gpu_writer import CustomWriter
from mattertune.wrappers.property_predictor import _create_trainer, _atoms_list_to_dataloader
from mattertune.util import is_rank_zero

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def load_partitioned_atoms(pkl_path: str) -> list[Atoms]:
    """从 pickle 文件加载 partitioned_atoms_list"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def write_predictions(preds: list[dict[str, np.ndarray]], out_path: str):
    """把 predictions list 序列化到 pickle"""
    with open(out_path, "wb") as f:
        pickle.dump(preds, f)


def run_inference(model, trainer, partitioned_atoms_list, args) -> list[dict[str, np.ndarray]]:
    """复用你原来的推理流程，返回一个 list of dict"""
    # 构造 dataloader
    num_devices = len(args.devices)
    partitions_per_rank = len(partitioned_atoms_list) // num_devices
    total_cores = os.cpu_count() or 1
    num_workers = min(partitions_per_rank, total_cores // num_devices)
    dl = _atoms_list_to_dataloader(
        partitioned_atoms_list, model,
        batch_size=args.batch_size,
        num_workers=args.num_workers or num_workers,
    )
    # 推理
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull):
            trainer.predict(model=model, dataloaders=dl)
    # barrier + gather
    torch.distributed.barrier()
    if is_rank_zero():
        raw_preds = trainer.callbacks[0].gather_all_predictions()
        trainer.callbacks[0].cleanup()
        # 把 torch.Tensor 转 numpy
        return [
            {k: v.detach().cpu().numpy() for k, v in p.items()}
            for p in raw_preds
        ]
    else:
        return []  # 非 0 号进程不返回结果


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_path",  required=True)
    parser.add_argument("--devices",    type=int, nargs="*", default=[0])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers",type=int, default=None)
    parser.add_argument("--granularity",type=int, default=4)
    args = parser.parse_args()

    # ——1—— 加载模型
    if "jmp" in args.ckpt_path:
        model = JMPBackboneModule.load_from_checkpoint(args.ckpt_path)
    elif "mattersim" in args.ckpt_path:
        model = MatterSimM3GNetBackboneModule.load_from_checkpoint(args.ckpt_path)
    elif "orb" in args.ckpt_path:
        model = ORBBackboneModule.load_from_checkpoint(args.ckpt_path)
    elif "eqv2" in args.ckpt_path:
        model = EqV2BackboneModule.load_from_checkpoint(args.ckpt_path)
    else:
        raise ValueError("Unsupported model type")
    model.hparams.using_partition = True
    # 禁用多余 head …（同原代码）
    # …

    # Lightning Trainer
    writer = CustomWriter(write_interval="epoch")
    trainer = _create_trainer(
        {
            "accelerator": "gpu",
            "devices": args.devices,
            "precision": "32",
            "strategy": "ddp",
            "inference_mode": False,
            "enable_progress_bar": False,
            "barebones": True,
            "callbacks": [writer],
            "logger": False,
        },
        model
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ——2—— 监控循环
    print("启动监控，输入目录：", args.input_dir)
    try:
        while True:
            # 退出信号
            if os.path.exists(os.path.join(args.input_dir, "exit.signal")):
                print("收到 exit.signal，退出。")
                break

            for fname in os.listdir(args.input_dir):
                if not fname.endswith(".ready"):
                    continue
                job_name = fname[:-6]  # 去掉 ".ready"
                pkl_in   = os.path.join(args.input_dir, job_name + ".pkl")
                ready_f  = os.path.join(args.input_dir, fname)
                # 1) 载入
                partitions = load_partitioned_atoms(pkl_in)
                # 2) 推理
                preds = run_inference(model, trainer, partitions, args)
                # 3) 写回
                tmp_out = os.path.join(args.output_dir, job_name + ".pred.tmp")
                final_out = os.path.join(args.output_dir, job_name + ".pred.pkl")
                write_predictions(preds, tmp_out)
                os.replace(tmp_out, final_out)          # 原子重命名
                open(os.path.join(args.output_dir, job_name + ".done"), "w").close()
                # 4) 清理 ready 文件，避免重复
                os.remove(ready_f)

            time.sleep(0.5)  # 轮询间隔，可调
    except KeyboardInterrupt:
        print("用户中断，退出。")


if __name__ == "__main__":
    main()


torchrun --nproc_per_node 8 --nnodes 1