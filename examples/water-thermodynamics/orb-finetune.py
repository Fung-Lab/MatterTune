from __future__ import annotations

import logging
from pathlib import Path

import nshutils as nu
from lightning.pytorch.strategies import DDPStrategy

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig

logging.basicConfig(level=logging.DEBUG)
nu.pretty()


def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()

        ## Model Hyperparameters
        hparams.model = MC.ORBBackboneConfig.draft()
        hparams.model.pretrained_model = args_dict["model_name"]
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.optimizer = MC.AdamWConfig(lr=args_dict["lr"])
        hparams.model.lr_scheduler = MC.CosineAnnealingLRConfig(
            T_max=args_dict["max_epochs"]
        )

        # Add model properties
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(loss=MC.MAELossConfig(), loss_coefficient=1.0)
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MAELossConfig(), conservative=False, loss_coefficient=100.0
        )
        hparams.model.properties.append(forces)

        ## Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.XYZDatasetConfig.draft()
        hparams.data.dataset.src = Path(args_dict["xyz_path"])
        hparams.data.train_split = args_dict["train_split"]
        hparams.data.batch_size = args_dict["batch_size"]
        hparams.data.num_workers = 0

        ## Add Normalization for Energy
        hparams.model.normalizers = {
            "energy": [
                MC.PerAtomReferencingNormalizerConfig(
                    per_atom_references=Path("./data/{}-energy_reference.json".format(args_dict["xyz_path"].split("/")[-1].split(".")[0]))
                )
            ]
        }

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
            monitor=f"val/forces_mae", patience=200, mode="min"
        )

        # Configure Model Checkpoint
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename="orb-best",
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Examples",
                name="ORB-Water",
            )
        ]

        # Additional trainer settings
        hparams.trainer.additional_trainer_kwargs = {
            "inference_mode": False,
            "strategy": DDPStrategy(static_graph=True, find_unused_parameters=True),  #
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    model, trainer = MatterTuner(mt_config).tune()
    # trainer.save_checkpoint("finetuned.ckpt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="orb-v2")
    parser.add_argument("--xyz_path", type=str, default="./data/water_ef.xyz")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--devices", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
