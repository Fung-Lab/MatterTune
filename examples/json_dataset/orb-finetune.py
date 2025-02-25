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

        # Add property
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(
            loss=MC.MAELossConfig(),
            loss_coefficient=0.01,
            # name=args_dict["task"],
            # dtype="float",
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MAELossConfig(), conservative=False, loss_coefficient=50.0
        )
        hparams.model.properties.append(forces)
        stress = MC.StressesPropertyConfig(
            loss=MC.MAELossConfig(), loss_coefficient=50.0, conservative=False
        )
        hparams.model.properties.append(stress)

        ## Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.JSONDatasetConfig.draft()
        tasks = {
            "energy": args_dict["energy_attr"],
            "forces": args_dict["forces_attr"],
            "stress": args_dict["stress_attr"],
        }
        hparams.data.dataset.tasks = tasks
        hparams.data.dataset.src = args_dict["data_src"]
        hparams.data.train_split = args_dict["train_split"]
        hparams.data.validation_split = args_dict["validation_split"]
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
            monitor=f"val/total_loss", patience=200, mode="min"
        )

        # Configure Model Checkpoint
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor=f"val/total_loss",
            dirpath=f"./checkpoints-{args_dict['task']}",
            filename="orb-best",
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Examples",
                name=f"ORB-Matbench-{args_dict['task']}",
                offline=False,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-src", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--energy-attr", type=str, default="y")
    parser.add_argument("--forces-attr", type=str, default="forces")
    parser.add_argument("--stress-attr", type=str, default="stress")
    parser.add_argument("--model_name", type=str, default="orb-v2")
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--validation_split", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
