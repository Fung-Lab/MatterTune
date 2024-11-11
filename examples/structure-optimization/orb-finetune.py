from __future__ import annotations

import logging
import os
from pathlib import Path

import nshutils as nu
import pytorch_lightning as pl
import rich
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
        hparams.model = MC.ORBBackboneConfig.draft()
        hparams.model.pretrained_model = args_dict["model_name"]
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.optimizer = MC.AdamWConfig(lr=args_dict["lr"])

        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(loss=MC.MAELossConfig(), loss_coefficient=1.0)
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MAELossConfig(), conservative=False, loss_coefficient=1.0
        )
        hparams.model.properties.append(forces)
        stresses = MC.StressesPropertyConfig(
            loss=MC.MAELossConfig(), conservative=False, loss_coefficient=1.0
        )  ## Here we used gradient-based stress prediction, but it is not used in the loss
        hparams.model.properties.append(stresses)

        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.train = MC.MPTrajDatasetConfig.draft()
        hparams.data.train.split = "train"
        hparams.data.train.elements = ["Zn", "Mn", "O"]
        hparams.data.validation = MC.MPTrajDatasetConfig.draft()
        hparams.data.train.split = "val"
        hparams.data.train.elements = ["Zn", "Mn", "O"]
        hparams.data.batch_size = args_dict["batch_size"]
        hparams.data.num_workers = 0

        ## Trainer Hyperparameters
        wandb_logger = WandbLogger(project="MatterTune-Examples", name="ORB-ZnMnO")
        checkpoint_callback = ModelCheckpoint(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename="orb-best",
            save_top_k=1,
            mode="min",
        )
        hparams.lightning_trainer_kwargs = {
            "max_epochs": args_dict["max_epochs"],
            "accelerator": "gpu",
            "devices": args_dict["devices"],
            "strategy": DDPStrategy(static_graph=True, find_unused_parameters=True),
            "gradient_clip_algorithm": "value",
            "gradient_clip_val": 1.0,
            "precision": "bf16",
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
    parser.add_argument("--model_name", type=str, default="orb-v2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--devices", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
