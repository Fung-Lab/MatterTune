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

import mattertune
import mattertune.backbones
import mattertune.configs as MC
from mattertune import MatterTuner

logging.basicConfig(level=logging.DEBUG)

nu.pretty()


def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()

        ## Model Hyperparameters
        hparams.model = MC.M3GNetBackboneConfig.draft()
        hparams.model.graph_computer = MC.M3GNetGraphComputerConfig.draft()
        hparams.model.graph_computer.pre_compute_line_graph = True
        hparams.model.ckpt_path = Path(args_dict["ckpt_path"])
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.optimizer = MC.AdamWConfig(lr=args_dict["lr"])

        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(loss=MC.MAELossConfig(), loss_coefficient=1.0)
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MAELossConfig(), conservative=True, loss_coefficient=1.0
        )
        hparams.model.properties.append(forces)
        stresses = MC.StressesPropertyConfig(
            loss=MC.MAELossConfig(),
            conservative=True,
            loss_coefficient=0.0,
        )  ## Here we used gradient-based stress prediction, but it is not used in the loss. And we scaled the stress by 160.21766208 to convert from eV/Angstrom^3 to GPa.
        hparams.model.properties.append(stresses)

        ## Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.XYZDatasetConfig.draft()
        hparams.data.dataset.src = Path(args_dict["xyz_path"])
        hparams.data.train_split = args_dict["train_split"]
        hparams.data.batch_size = args_dict["batch_size"]

        ## Trainer Hyperparameters
        wandb_logger = WandbLogger(
            project="MatterTune-Examples",
            name="M3GNet-Water",
            mode = "online",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename="m3gnet-best",
            save_top_k=1,
            mode="min",
        )
        hparams.lightning_trainer_kwargs = {
            "max_epochs": args_dict["max_epochs"],
            "accelerator": "gpu",
            "devices": args_dict["devices"],
            "strategy": "ddp",
            "gradient_clip_algorithm": "value",
            "gradient_clip_val": 1.0,
            "precision": "32",
            "inference_mode": False,
            "logger": [wandb_logger],
            "callbacks": [checkpoint_callback],
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    model = MatterTuner(mt_config).tune()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="M3GNet-MP-2021.2.8-PES")
    parser.add_argument("--xyz_path", type=str, default="./data/water_ef.xyz")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
