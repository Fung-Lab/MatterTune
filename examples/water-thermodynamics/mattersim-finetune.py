from __future__ import annotations

import logging
from pathlib import Path

import nshutils as nu

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig

logging.basicConfig(level=logging.DEBUG)
nu.pretty()


def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()

        ## Model Hyperparameters
        hparams.model = MC.MatterSimBackboneConfig.draft()
        hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        hparams.model.pretrained_model = args_dict["model_name"]
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.freeze_backbone = args_dict["freeze_backbone"]
        hparams.model.optimizer = MC.AdamWConfig(lr=args_dict["lr"])
        hparams.model.lr_scheduler = MC.StepLRConfig(
            step_size=args_dict["steplr_step_size"], gamma=args_dict["steplr_gamma"]
        )

        # Add model properties
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(), loss_coefficient=1.0 / (192**2)
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(), conservative=True, loss_coefficient=1.0
        )
        hparams.model.properties.append(forces)

        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.train = MC.XYZDatasetConfig.draft()
        hparams.data.train.src = "./data/train_water_1000_eVAng.xyz"
        hparams.data.train.down_sample = args_dict["train_down_sample"]
        hparams.data.train.down_sample_refill = args_dict["down_sample_refill"]
        hparams.data.validation = MC.XYZDatasetConfig.draft()
        hparams.data.validation.src = "./data/val_water_1000_eVAng.xyz"
        hparams.data.batch_size = args_dict["batch_size"]

        ## Add Normalization for Energy
        hparams.model.normalizers = {
            "energy": [
                MC.PerAtomReferencingNormalizerConfig(
                    per_atom_references=Path(
                        "./data/water_1000_eVAng-energy_reference.json"
                    )
                )
            ]
        }

        ## Trainer Hyperparameters
        hparams.trainer = MC.TrainerConfig.draft()
        hparams.trainer.max_epochs = args_dict["max_epochs"]
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.strategy = "ddp"
        hparams.trainer.gradient_clip_algorithm = "norm"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "32"

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/forces_mae", patience=50, mode="min"
        )

        # Configure Model Checkpoint
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename=args_dict["model_name"]
            + "-best"
            + str(args_dict["down_sample_refill"])
            + str(args_dict["steplr_gamma"]),
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Examples",
                name=args_dict["model_name"] + "-Water",
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
    # trainer.save_checkpoint("finetuned.ckpt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MatterSim-v1.0.0-1M")
    parser.add_argument("--freeze_backbone", type=bool, default=False)
    parser.add_argument("--train_down_sample", type=int, default=30)
    parser.add_argument("--down_sample_refill", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--devices", type=int, nargs="+", default=[3])
    parser.add_argument("--steplr_step_size", type=int, default=10)
    parser.add_argument("--steplr_gamma", type=float, default=0.9)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
