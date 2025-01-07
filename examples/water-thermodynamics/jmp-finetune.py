from __future__ import annotations

import logging
from pathlib import Path

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
        hparams.model = MC.JMPBackboneConfig.draft()
        hparams.model.graph_computer = MC.JMPGraphComputerConfig.draft()
        hparams.model.graph_computer.pbc = True
        hparams.model.ckpt_path = Path(args_dict["ckpt_path"])
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.optimizer = MC.AdamWConfig(lr=args_dict["lr"])
        hparams.model.lr_scheduler = MC.CosineAnnealingLRConfig(
            T_max=args_dict["max_epochs"],
            eta_min=1.0e-8,
        )
        hparams.model.freeze_backbone = args_dict["freeze_backbone"]
        hparams.model.reset_output_heads = True

        # Add model properties
        hparams.model.properties = []

        # Add energy property
        energy = MC.EnergyPropertyConfig(
            loss=MC.HuberLossConfig(), loss_coefficient=1.0
        )
        hparams.model.properties.append(energy)

        # Add forces property
        forces = MC.ForcesPropertyConfig(
            loss=MC.HuberLossConfig(),
            conservative=args_dict["conservative"],
            loss_coefficient=10.0,
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
            filename=args_dict["ckpt_path"].split("/")[-1].split(".")[0]
            + "-best"
            + str(args_dict["down_sample_refill"]),
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Examples", name="JMP-Water", offline=False
            )
        ]

        # Additional trainer settings that need special handling
        hparams.trainer.additional_trainer_kwargs = {
            "inference_mode": False,
            "strategy": DDPStrategy(find_unused_parameters=True),  # Special DDP config
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    model, trainer = MatterTuner(mt_config).tune()
    # trainer.save_checkpoint("finetuned.ckpt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/net/csefiles/coc-fung-cluster/lingyu/checkpoints/jmp-s.pt",
    )
    parser.add_argument("--conservative", type=bool, default=True)
    parser.add_argument("--freeze_backbone", type=bool, default=False)
    parser.add_argument("--train_down_sample", type=int, default=30)
    parser.add_argument("--down_sample_refill", type=bool, default=False)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
