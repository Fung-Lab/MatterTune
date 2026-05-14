from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.data import MatterTuneDataModule
from mattertune.configs import WandbLoggerConfig

from component_reference import load_component_reference
from component_training import attach_component_reference
from train import (  # noqa: E402
    DATA_ROOT,
    DEFAULT_TEST_FILE,
    normalize_devices,
)
from evaluate_checkpoint import evaluate_checkpoint  # noqa: E402


DEFAULT_TRAIN_FILE = DATA_ROOT / "Li_system_train_with_del.xyz"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "local_runs" / "elec-Li-withDel-03"
DEFAULT_COMPONENT_REFERENCE = (
    DEFAULT_OUTPUT_ROOT
    / "references"
    / "Li_system_train_with_del-MatterSim-v1.0.0-1M-component-residual-ridge-alpha1.0.json"
)


def _json_sanitize(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(value) for value in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def build_component_config(args: argparse.Namespace):
    hparams = MC.MatterTunerConfig.draft()

    hparams.model = MC.MatterSimBackboneConfig.draft()
    hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
    hparams.model.pretrained_model = args.model_name
    hparams.model.ignore_gpu_batch_transform_error = True
    hparams.model.freeze_backbone = False
    hparams.model.reset_output_heads = args.reset_output_heads

    hparams.model.optimizer = MC.AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=args.weight_decay,
    )
    hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
        mode="min",
        monitor=args.monitor,
        factor=0.8,
        patience=args.lr_patience,
        min_lr=1e-8,
    )
    hparams.model.properties = [
        MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(),
            loss_coefficient=args.e_loss_weight,
        ),
        MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(),
            loss_coefficient=args.f_loss_weight,
            conservative=True,
        ),
    ]
    hparams.model.normalizers = {}

    hparams.data = MC.AutoSplitDataModuleConfig.draft()
    hparams.data.dataset = MC.XYZDatasetConfig.draft()
    hparams.data.dataset.src = str(args.train_file)
    hparams.data.train_split = args.train_split
    hparams.data.shuffle = True
    hparams.data.shuffle_seed = args.shuffle_seed
    hparams.data.batch_size = args.batch_size
    hparams.data.pin_memory = False
    hparams.data.num_workers = args.num_workers

    hparams.trainer = MC.TrainerConfig.draft()
    hparams.trainer.max_epochs = args.max_epochs
    hparams.trainer.accelerator = args.accelerator
    hparams.trainer.devices = args.devices
    if len(args.devices) > 1:
        hparams.trainer.strategy = "ddp"
    hparams.trainer.gradient_clip_algorithm = "norm"
    hparams.trainer.gradient_clip_val = args.gradient_clip_val
    hparams.trainer.precision = "32"
    hparams.trainer.ema = MC.EMAConfig(decay=args.ema_decay)
    hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
        monitor=args.monitor,
        patience=args.patience,
        mode="min",
        min_delta=1.0e-5,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{args.model_name}-withDel-componentRef-best"
    ckpt_path = checkpoint_dir / f"{ckpt_name}.ckpt"
    if ckpt_path.exists():
        os.remove(ckpt_path)
    hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
        monitor=args.monitor,
        dirpath=str(checkpoint_dir),
        filename=ckpt_name,
        save_top_k=1,
        mode="min",
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-mattersim-withDel-03"
    config_snapshot = {
        "cli": _json_sanitize(vars(args)),
        "component_reference": str(args.component_reference),
        "mattertune": json.loads(hparams.model_dump_json()),
    }
    if args.logger == "wandb":
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project=args.wandb_project,
                name=run_name,
                offline=args.wandb_offline,
                save_dir=str(log_dir),
                additional_init_parameters={"config": config_snapshot},
            )
        ]
    elif args.logger == "csv":
        hparams.trainer.loggers = [
            MC.CSVLoggerConfig(save_dir=str(log_dir), name="lightning_logs")
        ]
    else:
        raise ValueError(f"Unsupported logger: {args.logger}")

    additional_trainer_kwargs = {"inference_mode": False}
    if args.limit_train_batches is not None:
        additional_trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        additional_trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    hparams.trainer.additional_trainer_kwargs = additional_trainer_kwargs

    return hparams.finalize(strict=False)


def fit_component_reference_model(args: argparse.Namespace) -> tuple[Any, Trainer]:
    config = build_component_config(args)
    config.model.ensure_dependencies()

    model = config.model.create_model()
    feature_names, coefficients = load_component_reference(args.component_reference)
    attach_component_reference(
        model,
        feature_names=feature_names,
        coefficients=coefficients,
        per_atom_energy_normalize=args.per_atom_energy_normalize,
    )

    datamodule = MatterTuneDataModule(config.data)
    trainer_kwargs = config.trainer._to_lightning_kwargs()
    if model.requires_disabled_inference_mode():
        trainer_kwargs["inference_mode"] = False
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule)
    return model, trainer


def main(args: argparse.Namespace) -> None:
    _, trainer = fit_component_reference_model(args)
    if args.skip_eval:
        print("skip_eval set; skipping test-set evaluation.")
        return

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    args.checkpoint = Path(best_ckpt_path)
    args.device = args.eval_device or f"cuda:{args.devices[0]}"
    args.metrics_name = "test_metrics.json"
    args.plot_name = "test_parity.png"
    args.predictions_name = "test_predictions.csv"
    args.gap_pairs_name = "test_gap_exact_pairs.csv"
    args.no_plot = False
    args.coordinate_match_decimals = 6
    args.max_force_plot_points = args.max_force_plot_points
    args.seed = args.eval_seed
    metrics = evaluate_checkpoint(args)
    flat_metrics = {
        f"test_eval/{group}/{name}": value
        for group, group_metrics in metrics.items()
        if isinstance(group_metrics, dict)
        for name, value in group_metrics.items()
        if isinstance(value, (int, float))
    }
    for logger in trainer.loggers:
        logger.log_metrics(flat_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="MatterSim-v1.0.0-1M")
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--component_reference", type=Path, default=DEFAULT_COMPONENT_REFERENCE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--e_loss_weight", type=float, default=200.0)
    parser.add_argument("--f_loss_weight", type=float, default=1.0)
    parser.add_argument("--monitor", default="val/total_loss")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--gradient_clip_val", type=float, default=2.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--logger", choices=("wandb", "csv"), default="wandb")
    parser.add_argument("--wandb_project", default="MatterTune-Electrolyte-Li-withDel-03")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--eval_device", default="")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--max_eval_structures", type=int, default=None)
    parser.add_argument("--max_force_plot_points", type=int, default=200000)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--reset_output_heads", action="store_true")
    parser.add_argument("--no_per_atom_energy_normalize", action="store_true")
    args = parser.parse_args()
    args.devices = normalize_devices(args.devices)
    args.per_atom_energy_normalize = not args.no_per_atom_energy_normalize
    args.energy_reference = args.component_reference

    run_name = args.wandb_name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-mattersim-withDel-03"
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / run_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(args.output_dir) / "checkpoints"
    if args.log_dir is None:
        args.log_dir = Path(args.output_dir) / "logs"

    for required in (args.train_file, args.test_file, args.component_reference):
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    return args


if __name__ == "__main__":
    main(parse_args())
