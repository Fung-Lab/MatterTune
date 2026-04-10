from __future__ import annotations

import logging
from pathlib import Path
import rich
import os

from lightning.pytorch.strategies import DDPStrategy

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig


def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()
        hparams.model = MC.MatterSimBackboneConfig.draft()
        hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        hparams.model.pretrained_model = args_dict["model_type"]
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.freeze_backbone = False
        hparams.model.reset_output_heads = True
        hparams.model.optimizer = MC.AdamWConfig(
            lr=args_dict["lr"],
            amsgrad=False,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.1,
        )
        hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
            mode="min",
            monitor=f"val/total_loss",
            factor=0.8,
            patience=5,
            min_lr=1e-8,
        )

        # Add model properties
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(), loss_coefficient=1.0
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(), conservative=True, loss_coefficient=1.0
        )
        hparams.model.properties.append(forces)

        # Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.XYZDatasetConfig.draft()
        hparams.data.dataset.src = "/net/csefiles/coc-fung-cluster/lingyu/electrolyte/Li-system-train-ase.xyz"
        hparams.data.train_split = 0.9
        hparams.data.shuffle = True
        hparams.data.shuffle_seed = 42
        hparams.data.batch_size = args_dict["batch_size"]
        hparams.data.pin_memory = False

        # Add Normalization for Energy
        hparams.model.normalizers = {
            "energy": [
                MC.PerAtomReferencingNormalizerConfig(
                    per_atom_references=Path(
                        "./data/Li-system-train-ase-energy_reference.json")
                ),
                MC.PerAtomNormalizerConfig(),
            ]
        }

        # Trainer Hyperparameters
        hparams.trainer = MC.TrainerConfig.draft()
        hparams.trainer.max_epochs = 2000
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.strategy = DDPStrategy()
        hparams.trainer.gradient_clip_algorithm = "norm"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "32"

        # Configure EMA
        hparams.trainer.ema = MC.EMAConfig(decay=0.99)

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/total_loss", patience=50, mode="min", min_delta=1e-4
        )

        # Configure Model Checkpoint
        ckpt_name = args_dict["model_type"] + "-best"
        if os.path.exists(f"./checkpoints/{ckpt_name}.ckpt"):
            os.remove(f"./checkpoints/{ckpt_name}.ckpt")
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/total_loss",
            dirpath="./checkpoints",
            filename=ckpt_name,
            save_top_k=1,
            mode="min",
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Electrolyte",
                name=args_dict["model_type"]+"-Li-system"
            )
        ]

        # Additional trainer settings
        hparams.trainer.additional_trainer_kwargs = {
            "inference_mode": False,
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    model, trainer = MatterTuner(mt_config).tune()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str,
                        default="MatterSim-v1.0.0-1M")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
