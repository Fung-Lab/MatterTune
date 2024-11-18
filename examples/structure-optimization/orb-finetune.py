from __future__ import annotations

import logging

import nshutils as nu
from lightning.pytorch.strategies import DDPStrategy

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig

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

        # Add energy property
        energy = MC.EnergyPropertyConfig(loss=MC.MAELossConfig(), loss_coefficient=1.0)
        hparams.model.properties.append(energy)

        # Add forces property
        forces = MC.ForcesPropertyConfig(
            loss=MC.MAELossConfig(), conservative=False, loss_coefficient=5.0
        )
        hparams.model.properties.append(forces)

        # Add stresses property
        stresses = MC.StressesPropertyConfig(
            loss=MC.MAELossConfig(), conservative=False, loss_coefficient=1.0
        )  # Here we used gradient-based stress prediction, but it is not used in the loss
        hparams.model.properties.append(stresses)

        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.train = MC.MPTrajDatasetConfig.draft()
        hparams.data.train.split = "train"
        hparams.data.train.elements = ["Zn", "Mn", "O"]
        hparams.data.validation = MC.MPTrajDatasetConfig.draft()
        hparams.data.validation.split = "val"
        hparams.data.validation.elements = ["Zn", "Mn", "O"]
        hparams.data.batch_size = args_dict["batch_size"]
        hparams.data.num_workers = 0

        ## Trainer Hyperparameters
        hparams.trainer = MC.TrainerConfig.draft()
        hparams.trainer.max_epochs = args_dict["max_epochs"]
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.gradient_clip_algorithm = "value"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "bf16"

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor="val/forces_mae", patience=200, mode="min"
        )

        # Configure Model Checkpoint
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename="orb-best",
            save_top_k=1,
            mode="min",
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Examples", name="ORB-ZnMnO", offline=False
            )
        ]

        # Additional trainer settings that need special handling
        hparams.trainer.additional_trainer_kwargs = {
            "inference_mode": False,
            "strategy": DDPStrategy(
                static_graph=True, find_unused_parameters=True
            ),  # Special DDP config
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    model, trainer = MatterTuner(mt_config).tune()
    trainer.save_checkpoint("finetuned.ckpt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="orb-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
