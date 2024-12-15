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
        hparams.model.optimizer = MC.AdamWConfig(lr=args_dict["lr"])
        hparams.model.lr_scheduler = MC.CosineAnnealingLRConfig(
            T_max=args_dict["max_epochs"]
        )

        # Add model properties
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(loss=MC.MAELossConfig(), loss_coefficient=1.0)
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MAELossConfig(), conservative=True, loss_coefficient=10.0
        )
        hparams.model.properties.append(forces)

        ## Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.XYZDatasetConfig.draft()
        hparams.data.dataset.src = Path(args_dict["xyz_path"])
        hparams.data.train_split = args_dict["train_split"]
        hparams.data.batch_size = args_dict["batch_size"]

        ## Add Normalization for Energy
        hparams.model.normalizers = {
            "energy": [
                MC.PerAtomReferencingNormalizerConfig(
                    per_atom_references=Path(
                        "./data/{}-energy_reference.json".format(
                            args_dict["xyz_path"].split("/")[-1].split(".")[0]
                        )
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
        hparams.trainer.gradient_clip_algorithm = "value"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "32"

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/forces_mae", patience=200, mode="min"
        )

        # Configure Model Checkpoint
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename="m3gnet-best",
            save_top_k=1,
            mode="min",
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Examples",
                name="MatterSim-Water",
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
    parser.add_argument("--model_name", type=str, default="MatterSim-v1.0.0-5M")
    parser.add_argument("--xyz_path", type=str, default="./data/water_ef.xyz")
    parser.add_argument("--train_split", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
