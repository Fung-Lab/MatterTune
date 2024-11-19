from __future__ import annotations

import logging
from pathlib import Path

import mattertune.configs as MC
import nshutils as nu
from lightning.pytorch.strategies import DDPStrategy
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

        # Add property
        hparams.model.properties = []
        property = MC.GraphPropertyConfig(
            loss=MC.MAELossConfig(),
            loss_coefficient=1.0,
            reduction="mean",
            name=args_dict["task"],
            dtype="float",
        )
        hparams.model.properties.append(property)

        ## Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.MatbenchDatasetConfig.draft()
        hparams.data.dataset.task = args_dict["task"]
        hparams.data.dataset.fold_idx = 0
        hparams.data.train_split = args_dict["train_split"]
        hparams.data.batch_size = args_dict["batch_size"]

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
            monitor=f"val/{args_dict['task']}_mae", patience=200, mode="min"
        )

        # Configure Model Checkpoint
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor=f"val/{args_dict['task']}_mae",
            dirpath=f"./checkpoints-{args_dict['task']}",
            filename="jmp-best",
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Examples",
                name=f"JMP-Matbench-{args_dict['task']}",
                offline=False,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/net/csefiles/coc-fung-cluster/lingyu/checkpoints/jmp-s.pt",
    )
    parser.add_argument("--task", type=str, default="matbench_log_kvrh")
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--devices", type=int, nargs="+", default=[1, 3])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
