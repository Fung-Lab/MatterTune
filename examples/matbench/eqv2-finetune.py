from __future__ import annotations

import logging
import os
from pathlib import Path

import nshutils as nu
import nshconfig_extra as CE
import pytorch_lightning as pl
import rich
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
        hparams.model = MC.EqV2BackboneConfig.draft()
        hparams.model.checkpoint_path = CE.CachedPath(
            uri="hf://fairchem/OMAT24/eqV2_31M_mp.pt"
        )
        hparams.model.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig.draft()
        hparams.model.atoms_to_graph.radius = 8.0
        hparams.model.atoms_to_graph.max_num_neighbors = 20

        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.optimizer = MC.AdamWConfig(lr=args_dict["lr"])

        hparams.model.properties = []
        property = MC.GraphPropertyConfig(
            loss=MC.MAELossConfig(),
            loss_coefficient=1.0,
            reduction="mean",
            name=args_dict["property"],
            dtype="float",
        )
        hparams.model.properties.append(property)

        ## Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.XYZDatasetConfig.draft()
        hparams.data.dataset.src = Path(args_dict["xyz_path"])
        hparams.data.train_split = args_dict["train_split"]
        hparams.data.batch_size = args_dict["batch_size"]

        ## Trainer Hyperparameters
        wandb_logger = WandbLogger(project="MatterTune-Examples", name="JMP-Water")
        checkpoint_callback = ModelCheckpoint(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename="jmp-best",
            save_top_k=1,
            mode="min",
        )
        early_stopping = EarlyStopping(
            monitor="val/forces_mae", patience=200, mode="min"
        )
        hparams.lightning_trainer_kwargs = {
            "max_epochs": args_dict["max_epochs"],
            "accelerator": "gpu",
            "devices": args_dict["devices"],
            "strategy": "ddp",
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
    parser.add_argument("--model_name", type=str, default="orb-v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
