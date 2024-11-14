from __future__ import annotations

import logging
import os
from pathlib import Path

import nshutils as nu
import pytorch_lightning as pl
import rich
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import mattertune
import mattertune.backbones
import mattertune.configs as MC
from mattertune import MatterTuner

logging.basicConfig(level=logging.WARNING)

nu.pretty()


def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()

        ## Model Hyperparameters
        hparams.model = MC.JMPBackboneConfig.draft()
        hparams.model.graph_computer = MC.JMPGraphComputerConfig.draft()
        hparams.model.graph_computer.pbc = True
        hparams.model.ckpt_path = Path(args_dict["ckpt_path"])
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.optimizer = MC.AdamWConfig(lr=args_dict["lr"])

        hparams.model.properties = []
        property = MC.GraphPropertyConfig(
            loss=MC.MAELossConfig(),
            loss_coefficient=1.0,
            reduction="mean",
            name=args_dict["task"],
            dtype="float",
        )
        hparams.model.properties.append(property)

        ## Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.MatbenchDatasetConfig.draft()
        hparams.data.dataset.task = args_dict["task"]
        hparams.data.dataset.fold_idx = 0
        hparams.data.train_split = args_dict["train_split"]
        hparams.data.batch_size = args_dict["batch_size"]

        ## Trainer Hyperparameters
        wandb_logger = WandbLogger(
            project="MatterTune-Examples",
            name="JMP-Matbench-{}".format(args_dict["task"]),
            mode="online",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val/matbench_mp_gap_mae",
            dirpath="./checkpoints",
            filename="jmp-best",
            save_top_k=1,
            mode="min",
        )
        early_stopping = EarlyStopping(
            monitor="val/matbench_mp_gap_mae", patience=200, mode="min"
        )
        hparams.lightning_trainer_kwargs = {
            "max_epochs": args_dict["max_epochs"],
            "accelerator": "gpu",
            "devices": args_dict["devices"],
            "strategy": DDPStrategy(find_unused_parameters=True),
            "gradient_clip_algorithm": "value",
            "gradient_clip_val": 1.0,
            "precision": "bf16",
            "inference_mode": False,
            "logger": [wandb_logger],
            "callbacks": [checkpoint_callback, early_stopping],
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    model = MatterTuner(mt_config).tune()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/net/csefiles/coc-fung-cluster/lingyu/checkpoints/jmp-s.pt",
    )
    parser.add_argument("--task", type=str, default="matbench_mp_gap")
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--devices", type=int, nargs="+", default=[1, 3])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)