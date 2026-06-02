from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rich
import torch
from ase import Atoms
from ase.io import read
from rich.progress import track

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig
from mattertune.main import load_finetuned_checkpoint


EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100")
TEST_DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/electrolyte")
DEFAULT_TRAIN_FILE = DATA_ROOT / "Li_system_lambda_parent_del_pairs.xyz"
DEFAULT_TEST_FILE = TEST_DATA_ROOT / "Li_system_test_with_del.xyz"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "local_runs" / "03-train-pair-100"
MODEL_TYPES = ("mattersim", "orb", "uma")


def normalize_model_type(raw: str) -> str:
    model_type = raw.strip().lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type {raw!r}; expected one of {MODEL_TYPES}.")
    return model_type


def normalize_model_name(model_type: str, model_name: str) -> str:
    model_type = normalize_model_type(model_type)
    name = model_name.strip()
    if model_type == "orb":
        aliases = {
            "orbv3-omat-conservative-inf": "orb-v3-conservative-inf-omat",
            "orb-v3-omat-conservative-inf": "orb-v3-conservative-inf-omat",
            "orbv3-conservative-inf-omat": "orb-v3-conservative-inf-omat",
        }
        return aliases.get(name, name.replace("_", "-"))
    if model_type == "uma":
        aliases = {
            "uma-s1.1": "uma-s-1p1",
            "uma-s-1.1": "uma-s-1p1",
            "uma-s1p1": "uma-s-1p1",
            "uma-s1.2": "uma-s-1p2",
            "uma-s-1.2": "uma-s-1p2",
            "uma-s1p2": "uma-s-1p2",
        }
        return aliases.get(name, name)
    return name


def model_label(model_type: str, model_name: str) -> str:
    safe_name = normalize_model_name(model_type, model_name).replace("/", "_")
    return f"{normalize_model_type(model_type)}-{safe_name}"


def _json_sanitize(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def normalize_devices(raw_devices: list[int | str]) -> list[int]:
    devices: list[int] = []
    for item in raw_devices:
        if isinstance(item, int):
            devices.append(item)
            continue
        for piece in str(item).split(","):
            piece = piece.strip()
            if piece:
                devices.append(int(piece))
    if not devices:
        raise ValueError("At least one device must be provided.")
    return devices


def build_config(args: argparse.Namespace):
    hparams = MC.MatterTunerConfig.draft()

    args.model_type = normalize_model_type(args.model_type)
    args.model_name = normalize_model_name(args.model_type, args.model_name)

    if args.model_type == "mattersim":
        hparams.model = MC.MatterSimBackboneConfig.draft()
        hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        hparams.model.pretrained_model = args.model_name
    elif args.model_type == "orb":
        hparams.model = MC.ORBBackboneConfig.draft()
        hparams.model.pretrained_model = args.model_name
        hparams.model.system = MC.ORBSystemConfig(
            radius=args.graph_radius,
            max_num_neighbors=args.max_num_neighbors,
            edge_method=args.orb_edge_method or None,
        )
    elif args.model_type == "uma":
        hparams.model = MC.UMABackboneConfig.draft()
        hparams.model.model_name = args.model_name
        hparams.model.task_name = args.task_name
        hparams.model.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig(
            radius=args.graph_radius,
            max_num_neighbors=args.max_num_neighbors,
        )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
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

    hparams.data = MC.AutoSplitDataModuleConfig.draft()
    hparams.data.dataset = MC.XYZDatasetConfig.draft()
    hparams.data.dataset.src = str(args.train_file)
    hparams.data.train_split = args.train_split
    hparams.data.shuffle = True
    hparams.data.shuffle_seed = args.shuffle_seed
    hparams.data.batch_size = args.batch_size
    hparams.data.pin_memory = False
    hparams.data.num_workers = args.num_workers

    energy_normalizers = [
        MC.PerAtomReferencingNormalizerConfig(
            per_atom_references=Path(args.energy_reference)
        )
    ]
    if args.per_atom_energy_normalize:
        energy_normalizers.append(MC.PerAtomNormalizerConfig())
    hparams.model.normalizers = {"energy": energy_normalizers}

    hparams.trainer = MC.TrainerConfig.draft()
    hparams.trainer.max_epochs = args.max_epochs
    hparams.trainer.accelerator = args.accelerator
    hparams.trainer.devices = args.devices
    if len(args.devices) > 1:
        hparams.trainer.strategy = "ddp"
    hparams.trainer.gradient_clip_algorithm = "norm"
    hparams.trainer.gradient_clip_val = args.gradient_clip_val
    hparams.trainer.precision = "32"
    hparams.trainer.resume_checkpoint = args.resume_checkpoint
    hparams.trainer.ema = MC.EMAConfig(decay=args.ema_decay)
    hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
        monitor=args.monitor,
        patience=args.patience,
        mode="min",
        min_delta=1.0e-5,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{model_label(args.model_type, args.model_name)}-pair100-best"
    ckpt_path = checkpoint_dir / f"{ckpt_name}.ckpt"
    if ckpt_path.exists() and args.resume_checkpoint is None:
        ckpt_path.unlink()
    hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
        monitor=args.monitor,
        dirpath=str(checkpoint_dir),
        filename=ckpt_name,
        save_last=True,
        save_top_k=1,
        mode="min",
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_name or (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{model_label(args.model_type, args.model_name)}-pair100"
    )
    config_snapshot = {
        "cli": _json_sanitize(vars(args)),
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


def structure_group(atoms: Atoms) -> str:
    return "normal" if "config_type" in atoms.info else "deleted"


def summarize_errors(
    *,
    energy_gt: np.ndarray,
    energy_pred: np.ndarray,
    natoms: np.ndarray,
    forces_gt: list[np.ndarray],
    forces_pred: list[np.ndarray],
    groups: list[str],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    group_names = ["all"] + sorted(set(groups))
    for group_name in group_names:
        if group_name == "all":
            idx = np.arange(len(groups))
        else:
            idx = np.asarray([i for i, group in enumerate(groups) if group == group_name])
        if len(idx) == 0:
            continue

        e_err = energy_pred[idx] - energy_gt[idx]
        epa_err = energy_pred[idx] / natoms[idx] - energy_gt[idx] / natoms[idx]
        f_gt = np.vstack([forces_gt[i] for i in idx])
        f_pred = np.vstack([forces_pred[i] for i in idx])
        f_err = f_pred - f_gt

        metrics[group_name] = {
            "n_structures": int(len(idx)),
            "n_force_components": int(f_err.size),
            "energy_mae_eV": float(np.mean(np.abs(e_err))),
            "energy_rmse_eV": float(np.sqrt(np.mean(e_err**2))),
            "energy_bias_eV": float(np.mean(e_err)),
            "energy_per_atom_mae_eV": float(np.mean(np.abs(epa_err))),
            "energy_per_atom_rmse_eV": float(np.sqrt(np.mean(epa_err**2))),
            "energy_per_atom_bias_eV": float(np.mean(epa_err)),
            "force_component_mae_eV_A": float(np.mean(np.abs(f_err))),
            "force_component_rmse_eV_A": float(np.sqrt(np.mean(f_err**2))),
            "force_component_bias_eV_A": float(np.mean(f_err)),
        }
    return metrics


def save_parity_plot(
    output_path: Path,
    *,
    energy_gt: np.ndarray,
    energy_pred: np.ndarray,
    forces_gt: list[np.ndarray],
    forces_pred: list[np.ndarray],
    max_force_points: int,
    seed: int,
) -> None:
    f_gt = np.vstack(forces_gt).reshape(-1)
    f_pred = np.vstack(forces_pred).reshape(-1)
    if f_gt.size > max_force_points:
        rng = np.random.default_rng(seed)
        choice = rng.choice(f_gt.size, size=max_force_points, replace=False)
        f_gt = f_gt[choice]
        f_pred = f_pred[choice]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.scatter(energy_gt, energy_pred, s=8, alpha=0.55)
    emin = min(float(energy_gt.min()), float(energy_pred.min()))
    emax = max(float(energy_gt.max()), float(energy_pred.max()))
    ax.plot([emin, emax], [emin, emax], color="k", linewidth=1.0)
    ax.set_xlim(emin, emax)
    ax.set_ylim(emin, emax)
    ax.set_xlabel("DFT energy (eV)")
    ax.set_ylabel("MLIP energy (eV)")
    ax.set_title("Energy")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ax.scatter(f_gt, f_pred, s=1, alpha=0.25)
    fmin = min(float(f_gt.min()), float(f_pred.min()))
    fmax = max(float(f_gt.max()), float(f_pred.max()))
    ax.plot([fmin, fmax], [fmin, fmax], color="k", linewidth=1.0)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(fmin, fmax)
    ax.set_xlabel("DFT force (eV/A)")
    ax.set_ylabel("MLIP force (eV/A)")
    ax.set_title("Force components")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_checkpoint(args: argparse.Namespace, ckpt_path: str | Path) -> dict[str, dict[str, float | int]]:
    model = load_finetuned_checkpoint(str(ckpt_path))
    eval_device = args.eval_device or f"cuda:{args.devices[0]}"
    calc = model.ase_calculator(device=eval_device)

    atoms_list: list[Atoms] = read(args.test_file, index=":")  # type: ignore[assignment]
    if args.max_eval_structures is not None:
        atoms_list = atoms_list[: args.max_eval_structures]

    energy_gt: list[float] = []
    energy_pred: list[float] = []
    forces_gt: list[np.ndarray] = []
    forces_pred: list[np.ndarray] = []
    natoms: list[int] = []
    groups: list[str] = []

    for atoms in track(atoms_list, description="Evaluating test set"):
        gt_e = float(atoms.get_potential_energy())
        gt_f = np.asarray(atoms.get_forces(), dtype=np.float64)
        atoms_for_pred = atoms.copy()
        atoms_for_pred.calc = calc
        pred_e = float(atoms_for_pred.get_potential_energy())
        pred_f = np.asarray(atoms_for_pred.get_forces(), dtype=np.float64)

        energy_gt.append(gt_e)
        energy_pred.append(pred_e)
        forces_gt.append(gt_f)
        forces_pred.append(pred_f)
        natoms.append(len(atoms))
        groups.append(structure_group(atoms))

    metrics = summarize_errors(
        energy_gt=np.asarray(energy_gt, dtype=np.float64),
        energy_pred=np.asarray(energy_pred, dtype=np.float64),
        natoms=np.asarray(natoms, dtype=np.float64),
        forces_gt=forces_gt,
        forces_pred=forces_pred,
        groups=groups,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4, sort_keys=True)

    plot_path = output_dir / "test_parity.png"
    save_parity_plot(
        plot_path,
        energy_gt=np.asarray(energy_gt, dtype=np.float64),
        energy_pred=np.asarray(energy_pred, dtype=np.float64),
        forces_gt=forces_gt,
        forces_pred=forces_pred,
        max_force_points=args.max_force_plot_points,
        seed=args.eval_seed,
    )

    rich.print(f"Saved test metrics to {metrics_path}")
    rich.print(f"Saved parity plot to {plot_path}")
    rich.print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main(args: argparse.Namespace) -> None:
    config = build_config(args)
    _, trainer = MatterTuner(config).tune()

    if args.skip_eval:
        rich.print("skip_eval set; skipping test-set evaluation.")
        return

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    metrics = evaluate_checkpoint(args, best_ckpt_path)
    flat_metrics = {
        f"test_eval/{group}/{name}": value
        for group, group_metrics in metrics.items()
        for name, value in group_metrics.items()
        if isinstance(value, (int, float))
    }
    for logger in trainer.loggers:
        logger.log_metrics(flat_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=MODEL_TYPES, default="mattersim")
    parser.add_argument("--model_name", default="MatterSim-v1.0.0-1M")
    parser.add_argument(
        "--task_name",
        default="omat",
        help="UMA task name, e.g. omat, omol, oc20, odac, or omc. Ignored by other backbones.",
    )
    parser.add_argument("--graph_radius", type=float, default=6.0)
    parser.add_argument("--max_num_neighbors", type=int, default=120)
    parser.add_argument(
        "--orb_edge_method",
        choices=("knn_brute_force", "knn_scipy", "knn_cuml_brute", "knn_cuml_rbc", "knn_alchemi"),
        default=None,
        help=(
            "Optional ORB graph edge-construction method. For CPU featurization, "
            "knn_scipy avoids nvalchemiops/Warp CUDA initialization noise."
        ),
    )
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--energy_reference", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--e_loss_weight", type=float, default=200.0)
    parser.add_argument("--f_loss_weight", type=float, default=20.0)
    parser.add_argument("--monitor", default="val/total_loss")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--gradient_clip_val", type=float, default=2.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--logger", choices=("wandb", "csv"), default="wandb")
    parser.add_argument("--wandb_project", default="MatterTune-Electrolyte-Li-pair-100")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--eval_device", default="")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--max_eval_structures", type=int, default=None)
    parser.add_argument("--max_force_plot_points", type=int, default=200000)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument(
        "--reset_output_heads",
        action="store_true",
        help=(
            "Reset output heads before finetuning. Default is false so the "
            "energy reference remains aligned with the raw finetuning head."
        ),
    )
    parser.add_argument(
        "--no_per_atom_energy_normalize",
        action="store_true",
        help="Disable energy loss normalization by number of atoms.",
    )
    args = parser.parse_args()
    args.model_type = normalize_model_type(args.model_type)
    args.model_name = normalize_model_name(args.model_type, args.model_name)
    args.devices = normalize_devices(args.devices)
    args.per_atom_energy_normalize = not args.no_per_atom_energy_normalize

    run_name = args.wandb_name or (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{model_label(args.model_type, args.model_name)}-pair100"
    )
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / run_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(args.output_dir) / "checkpoints"
    if args.log_dir is None:
        args.log_dir = Path(args.output_dir) / "logs"

    required_paths = [args.train_file, args.test_file, args.energy_reference]
    if args.resume_checkpoint is not None:
        required_paths.append(args.resume_checkpoint)
    for required in required_paths:
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    return args


if __name__ == "__main__":
    main(parse_args())
