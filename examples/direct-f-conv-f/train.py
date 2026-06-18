from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rich
import torch
from ase import Atoms
from ase.io import iread, read
from rich.progress import track

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig
from mattertune.main import load_finetuned_checkpoint
from mattertune.util import set_global_random_seed


EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(
    "/nethome/lkong88/workspace/Distill-New/AFMDistill-Old/examples/distill_from_md/data"
)
DEFAULT_TRAIN_FILE = DATA_ROOT / "h2o_1593_train_25.xyz"
DEFAULT_VAL_FILE = DATA_ROOT / "h2o_1593_val_5.xyz"
DEFAULT_TEST_FILE = DATA_ROOT / "h2o_1593_test_1563.xyz"
DEFAULT_WORK_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Direct-F-Conv-F")
DEFAULT_OUTPUT_ROOT = DEFAULT_WORK_ROOT / "runs"
DEFAULT_REFERENCE_ROOT = DEFAULT_WORK_ROOT / "references"
FORCE_MODES = ("direct", "conservative")


def normalize_force_mode(raw: str) -> str:
    force_mode = raw.strip().lower().replace("_", "-")
    aliases = {
        "direct-force": "direct",
        "direct-forces": "direct",
        "nonconservative": "direct",
        "non-conservative": "direct",
        "conservative-force": "conservative",
        "conservative-forces": "conservative",
    }
    force_mode = aliases.get(force_mode, force_mode)
    if force_mode not in FORCE_MODES:
        raise ValueError(f"Unsupported force_mode {raw!r}; expected one of {FORCE_MODES}.")
    return force_mode


def normalize_model_name(raw: str) -> str:
    name = raw.strip()
    aliases = {
        "uma-s1.1": "uma-s-1p1",
        "uma-s-1.1": "uma-s-1p1",
        "uma-s1p1": "uma-s-1p1",
        "uma-s-1p1": "uma-s-1p1",
        "uma-s1.2": "uma-s-1p2",
        "uma-s-1.2": "uma-s-1p2",
        "uma-s1p2": "uma-s-1p2",
        "uma-s-1p2": "uma-s-1p2",
    }
    return aliases.get(name, name)


def normalize_devices(raw_devices: list[int | str]) -> list[int] | str:
    if len(raw_devices) == 1:
        token = str(raw_devices[0]).strip().lower()
        if token in {"all", "auto"}:
            return token

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


def use_ddp(devices: list[int] | str) -> bool:
    if isinstance(devices, str):
        return devices == "all"
    return len(devices) > 1


def first_eval_device(devices: list[int] | str) -> str:
    if torch.cuda.is_available():
        if isinstance(devices, list):
            return f"cuda:{devices[0]}"
        return "cuda:0"
    return "cpu"


def infer_reference_device(devices: list[int] | str) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if isinstance(devices, list) and devices:
        return f"cuda:{devices[0]}"
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "cuda:0"

    best_index = None
    best_free = -1
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 2:
            continue
        try:
            index = int(pieces[0])
            free = int(pieces[1])
        except ValueError:
            continue
        if free > best_free:
            best_index = index
            best_free = free
    return f"cuda:{best_index}" if best_index is not None else "cuda:0"


def safe_label(value: object) -> str:
    label = str(value)
    for old, new in (("/", "_"), (" ", "_"), (".", "p"), (":", "-")):
        label = label.replace(old, new)
    return label


def experiment_label(args: argparse.Namespace) -> str:
    return "-".join(
        [
            "uma",
            safe_label(args.model_name),
            safe_label(args.task_name),
            args.force_mode,
        ]
    )


def reference_label(args: argparse.Namespace) -> str:
    return "-".join(
        [
            Path(args.train_file).stem,
            experiment_label(args),
            f"r{safe_label(args.graph_radius)}",
            f"n{args.max_num_neighbors}",
            f"seed{args.seed}",
            f"reset{int(args.reset_output_heads)}",
            "training-head-residual",
            args.reference_model,
            f"alpha{safe_label(args.ridge_alpha)}",
        ]
    )


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


def count_structures(path: Path) -> int:
    return sum(1 for _ in iread(path, index=":"))


def validate_count(path: Path, expected: int | None, label: str) -> None:
    if expected is None:
        return
    actual = count_structures(path)
    if actual != expected:
        raise ValueError(f"{label} split at {path} has {actual} structures; expected {expected}.")
    rich.print(f"{label} split: {actual} structures")


def composition_matrix(atoms_list: list[Atoms]) -> np.ndarray:
    max_z = max(max(Counter(atoms.numbers).keys()) for atoms in atoms_list)
    matrix = np.zeros((len(atoms_list), max_z + 1), dtype=np.float64)
    for row, atoms in enumerate(atoms_list):
        for z, count in Counter(atoms.numbers).items():
            matrix[row, z] = count
    return matrix


def fit_references(
    compositions: np.ndarray,
    residual_energies: np.ndarray,
    *,
    reference_model: str,
    ridge_alpha: float,
) -> dict[int, float]:
    if reference_model == "linear":
        coeffs = np.linalg.lstsq(compositions, residual_energies, rcond=None)[0]
    elif reference_model == "ridge":
        lhs = compositions.T @ compositions
        rhs = compositions.T @ residual_energies
        coeffs = np.linalg.solve(lhs + ridge_alpha * np.eye(lhs.shape[0]), rhs)
    else:
        raise ValueError(f"Unsupported reference model: {reference_model}")

    present = set(int(z) for row in compositions for z in np.flatnonzero(row))
    references = {int(z): float(coeffs[z]) for z in present if z != 0}
    return references


def _move_to_device(value: Any, device: torch.device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_move_to_device(item, device) for item in value))
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def create_training_energy_model(args: argparse.Namespace):
    model_config = MC.UMABackboneConfig.draft()
    model_config.model_name = args.model_name
    model_config.task_name = args.task_name
    model_config.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig(
        radius=args.graph_radius,
        max_num_neighbors=args.max_num_neighbors,
    )
    model_config.properties = [
        MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(),
            loss_coefficient=1.0,
        ),
        MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(),
            loss_coefficient=1.0,
            conservative=args.force_mode == "conservative",
        ),
    ]
    model_config.optimizer = MC.AdamWConfig(lr=1.0e-3)
    model_config.reset_output_heads = args.reset_output_heads
    model_config.freeze_backbone = False
    model_config.ignore_gpu_batch_transform_error = True
    model_config = model_config.finalize(strict=False)
    model_config.ensure_dependencies()
    model = model_config.create_model()
    model.eval()
    return model


def training_head_energies(atoms_list: list[Atoms], args: argparse.Namespace) -> np.ndarray:
    set_global_random_seed(args.seed)
    device = torch.device(args.reference_device)
    model = create_training_energy_model(args)
    model.to(device)

    energies: list[float] = []
    batch_size = max(1, int(args.reference_batch_size or args.batch_size))
    starts = range(0, len(atoms_list), batch_size)
    for start in track(starts, description="Training-head baseline energies"):
        chunk = atoms_list[start : start + batch_size]
        data_list = [
            model.cpu_data_transform(model.atoms_to_data(atoms, has_labels=False))
            for atoms in chunk
        ]
        batch = model.collate_fn(data_list)
        batch = _move_to_device(batch, device)
        with torch.enable_grad():
            output = model(batch, mode="predict", ignore_gpu_batch_transform_error=False)
        energy = output["predicted_properties"]["energy"].detach().cpu().reshape(-1)
        energies.extend(float(value) for value in energy)
    return np.asarray(energies, dtype=np.float64)


def fit_residual_energy_reference(args: argparse.Namespace) -> dict[int, float]:
    atoms_list = read(args.train_file, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError(f"No structures loaded from {args.train_file}")

    dft_energies = np.asarray(
        [float(atoms.get_potential_energy()) for atoms in atoms_list],
        dtype=np.float64,
    )
    baseline_energies = training_head_energies(atoms_list, args)
    residual = dft_energies - baseline_energies
    compositions = composition_matrix(atoms_list)

    references = fit_references(
        compositions,
        residual,
        reference_model=args.reference_model,
        ridge_alpha=args.ridge_alpha,
    )

    ref_vector = np.zeros(compositions.shape[1], dtype=np.float64)
    for z, value in references.items():
        ref_vector[z] = value
    fitted_residual = compositions @ ref_vector
    residual_after_ref = residual - fitted_residual

    args.energy_reference.parent.mkdir(parents=True, exist_ok=True)
    with args.energy_reference.open("w", encoding="utf-8") as handle:
        json.dump(references, handle, indent=4, sort_keys=True)

    summary = {
        "train_file": str(args.train_file),
        "frames": len(atoms_list),
        "model_type": "uma",
        "model_name": args.model_name,
        "task_name": args.task_name,
        "force_mode": args.force_mode,
        "reference_energy_source": "training_head",
        "graph_radius": args.graph_radius,
        "max_num_neighbors": args.max_num_neighbors,
        "seed": args.seed,
        "reset_output_heads": args.reset_output_heads,
        "reference_model": args.reference_model,
        "ridge_alpha": args.ridge_alpha,
        "mean_target_energy_eV": float(dft_energies.mean()),
        "mean_baseline_energy_eV": float(baseline_energies.mean()),
        "mean_residual_energy_eV": float(residual.mean()),
        "residual_after_reference_mae_eV": float(np.abs(residual_after_ref).mean()),
        "residual_after_reference_rmse_eV": float(np.sqrt(np.mean(residual_after_ref**2))),
        "references": references,
    }
    summary_path = args.energy_reference.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4, sort_keys=True)

    rich.print(f"Saved residual energy reference to {args.energy_reference}")
    rich.print(f"Saved residual energy reference summary to {summary_path}")
    rich.print(
        "Residual-after-reference MAE/RMSE: "
        f"{summary['residual_after_reference_mae_eV']:.8e} / "
        f"{summary['residual_after_reference_rmse_eV']:.8e} eV"
    )
    return references


def ensure_energy_reference(args: argparse.Namespace) -> None:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank is not None:
        if int(rank) == 0 and not args.energy_reference.is_file():
            fit_residual_energy_reference(args)
            return

        deadline = time.time() + args.reference_wait_timeout
        while not args.energy_reference.is_file() and time.time() < deadline:
            time.sleep(2.0)
        if not args.energy_reference.is_file():
            raise FileNotFoundError(args.energy_reference)
        return

    if args.refit_reference or not args.energy_reference.is_file():
        fit_residual_energy_reference(args)
    else:
        rich.print(f"Using existing residual energy reference: {args.energy_reference}")


def build_config(args: argparse.Namespace):
    hparams = MC.MatterTunerConfig.draft()

    hparams.model = MC.UMABackboneConfig.draft()
    hparams.model.model_name = args.model_name
    hparams.model.task_name = args.task_name
    hparams.model.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig(
        radius=args.graph_radius,
        max_num_neighbors=args.max_num_neighbors,
    )
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
        min_lr=args.min_lr,
    )
    hparams.model.properties = [
        MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(),
            loss_coefficient=args.e_loss_weight,
        ),
        MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(),
            loss_coefficient=args.f_loss_weight,
            conservative=args.force_mode == "conservative",
        ),
    ]

    hparams.data = MC.ManualSplitDataModuleConfig.draft()
    hparams.data.train = MC.XYZDatasetConfig.draft()
    hparams.data.train.src = str(args.train_file)
    hparams.data.validation = MC.XYZDatasetConfig.draft()
    hparams.data.validation.src = str(args.val_file)
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
    if use_ddp(args.devices):
        hparams.trainer.strategy = "ddp"
    hparams.trainer.gradient_clip_algorithm = "norm"
    hparams.trainer.gradient_clip_val = args.gradient_clip_val
    hparams.trainer.precision = args.precision
    hparams.trainer.resume_checkpoint = args.resume_checkpoint
    hparams.trainer.ema = MC.EMAConfig(decay=args.ema_decay)
    hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
        monitor=args.monitor,
        patience=args.patience,
        mode="min",
        min_delta=args.min_delta,
    )
    hparams.trainer.log_every_n_steps = args.log_every_n_steps

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{experiment_label(args)}-best"
    ckpt_path = args.checkpoint_dir / f"{ckpt_name}.ckpt"
    if ckpt_path.exists() and args.resume_checkpoint is None:
        ckpt_path.unlink()
    hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
        monitor=args.monitor,
        dirpath=str(args.checkpoint_dir),
        filename=ckpt_name,
        save_last=True,
        save_top_k=1,
        mode="min",
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "cli": _json_sanitize(vars(args)),
    }
    if args.logger == "wandb":
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project=args.wandb_project,
                name=args.run_name,
                offline=args.wandb_offline,
                save_dir=str(args.log_dir),
                additional_init_parameters={"config": config_snapshot},
            )
        ]
    elif args.logger == "csv":
        hparams.trainer.loggers = [
            MC.CSVLoggerConfig(save_dir=str(args.log_dir), name="lightning_logs")
        ]
    else:
        raise ValueError(f"Unsupported logger: {args.logger}")

    additional_trainer_kwargs: dict[str, Any] = {"inference_mode": False}
    if args.limit_train_batches is not None:
        additional_trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        additional_trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    hparams.trainer.additional_trainer_kwargs = additional_trainer_kwargs

    return hparams.finalize(strict=False)


def summarize_errors(
    *,
    energy_gt: np.ndarray,
    energy_pred: np.ndarray,
    natoms: np.ndarray,
    forces_gt: list[np.ndarray],
    forces_pred: list[np.ndarray],
) -> dict[str, float | int]:
    e_err = energy_pred - energy_gt
    epa_err = energy_pred / natoms - energy_gt / natoms
    f_gt = np.vstack(forces_gt)
    f_pred = np.vstack(forces_pred)
    f_err = f_pred - f_gt
    vector_norm_err = np.linalg.norm(f_err, axis=1)
    return {
        "n_structures": int(len(energy_gt)),
        "n_atoms_per_structure": int(natoms[0]) if len(set(natoms.tolist())) == 1 else -1,
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
        "force_vector_norm_mae_eV_A": float(np.mean(vector_norm_err)),
        "force_vector_norm_rmse_eV_A": float(np.sqrt(np.mean(vector_norm_err**2))),
    }


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
    ax.set_xlabel("Target energy (eV)")
    ax.set_ylabel("Predicted energy (eV)")
    ax.set_title("Energy")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ax.scatter(f_gt, f_pred, s=1, alpha=0.25)
    fmin = min(float(f_gt.min()), float(f_pred.min()))
    fmax = max(float(f_gt.max()), float(f_pred.max()))
    ax.plot([fmin, fmax], [fmin, fmax], color="k", linewidth=1.0)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(fmin, fmax)
    ax.set_xlabel("Target force (eV/A)")
    ax.set_ylabel("Predicted force (eV/A)")
    ax.set_title("Force components")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_checkpoint(args: argparse.Namespace, ckpt_path: str | Path) -> dict[str, float | int]:
    model = load_finetuned_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    calc = model.ase_calculator(device=args.eval_device or first_eval_device(args.devices))

    atoms_list = read(args.test_file, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if args.max_eval_structures is not None:
        atoms_list = atoms_list[: args.max_eval_structures]

    energy_gt: list[float] = []
    energy_pred: list[float] = []
    forces_gt: list[np.ndarray] = []
    forces_pred: list[np.ndarray] = []
    natoms: list[int] = []

    for atoms in track(atoms_list, description="Evaluating test set"):
        target_e = float(atoms.get_potential_energy())
        target_f = np.asarray(atoms.get_forces(), dtype=np.float64)
        atoms_for_pred = atoms.copy()
        atoms_for_pred.calc = calc
        pred_e = float(atoms_for_pred.get_potential_energy())
        pred_f = np.asarray(atoms_for_pred.get_forces(), dtype=np.float64)

        energy_gt.append(target_e)
        energy_pred.append(pred_e)
        forces_gt.append(target_f)
        forces_pred.append(pred_f)
        natoms.append(len(atoms))

    metrics = summarize_errors(
        energy_gt=np.asarray(energy_gt, dtype=np.float64),
        energy_pred=np.asarray(energy_pred, dtype=np.float64),
        natoms=np.asarray(natoms, dtype=np.float64),
        forces_gt=forces_gt,
        forces_pred=forces_pred,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4, sort_keys=True)

    plot_path = args.output_dir / "test_parity.png"
    save_parity_plot(
        plot_path,
        energy_gt=np.asarray(energy_gt, dtype=np.float64),
        energy_pred=np.asarray(energy_pred, dtype=np.float64),
        forces_gt=forces_gt,
        forces_pred=forces_pred,
        max_force_points=args.max_force_plot_points,
        seed=args.seed,
    )

    rich.print(f"Saved test metrics to {metrics_path}")
    rich.print(f"Saved parity plot to {plot_path}")
    rich.print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def write_run_config(args: argparse.Namespace, config: MC.MatterTunerConfig) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cli": _json_sanitize(vars(args)),
        "mattertune": json.loads(config.model_dump_json()),
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, sort_keys=True)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_global_random_seed(args.seed)
    ensure_energy_reference(args)
    set_global_random_seed(args.seed)
    config = build_config(args)
    write_run_config(args, config)

    _, trainer = MatterTuner(config).tune()
    if not trainer.is_global_zero:
        return

    if args.skip_eval:
        rich.print("skip_eval set; skipping test-set evaluation.")
        return

    checkpoint_callback = trainer.checkpoint_callback
    ckpt_path = getattr(checkpoint_callback, "best_model_path", "") or (
        args.checkpoint_dir / f"{experiment_label(args)}-best.ckpt"
    )
    metrics = evaluate_checkpoint(args, ckpt_path)
    flat_metrics = {
        f"test_eval/{name}": value
        for name, value in metrics.items()
        if isinstance(value, (int, float))
    }
    for logger in trainer.loggers:
        logger.log_metrics(flat_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="uma-s1p1")
    parser.add_argument("--task_name", default="omat")
    parser.add_argument(
        "--force_mode",
        default="conservative",
        help="direct/non-conservative force head or conservative forces from the energy head.",
    )
    parser.add_argument("--graph_radius", type=float, default=6.0)
    parser.add_argument("--max_num_neighbors", type=int, default=120)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val_file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--expected_train_count", type=int, default=25)
    parser.add_argument("--expected_val_count", type=int, default=5)
    parser.add_argument("--expected_test_count", type=int, default=1563)
    parser.add_argument("--no_validate_counts", action="store_true")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--energy_reference", type=Path, default=None)
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--refit_reference", action="store_true")
    parser.add_argument("--reference_device", default="")
    parser.add_argument("--reference_batch_size", type=int, default=None)
    parser.add_argument("--reference_model", choices=("linear", "ridge"), default="ridge")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--reference_wait_timeout", type=float, default=1800.0)
    parser.add_argument("--devices", nargs="+", default=["all"])
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--min_lr", type=float, default=1.0e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--e_loss_weight", type=float, default=1.0)
    parser.add_argument("--f_loss_weight", type=float, default=1.0)
    parser.add_argument("--monitor", default="val/forces_mae")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1.0e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--precision", default="32")
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logger", choices=("csv", "wandb"), default="csv")
    parser.add_argument("--wandb_project", default="MatterTune-H2O-UMA-direct-f-conv-f")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--eval_device", default="")
    parser.add_argument("--max_eval_structures", type=int, default=None)
    parser.add_argument("--max_force_plot_points", type=int, default=200000)
    parser.add_argument("--limit_train_batches", type=float, default=None)
    parser.add_argument("--limit_val_batches", type=float, default=None)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument(
        "--reset_output_heads",
        action="store_true",
        help=(
            "Reset UMA output heads before training. Default is false so the "
            "residual reference stays aligned with the initial finetuning head."
        ),
    )
    parser.add_argument(
        "--no_per_atom_energy_normalize",
        action="store_true",
        help="Disable per-atom scaling after residual energy referencing.",
    )

    args = parser.parse_args()
    args.model_name = normalize_model_name(args.model_name)
    args.force_mode = normalize_force_mode(args.force_mode)
    args.devices = normalize_devices(args.devices)
    if not args.reference_device:
        args.reference_device = infer_reference_device(args.devices)
    args.per_atom_energy_normalize = not args.no_per_atom_energy_normalize

    for path in (args.train_file, args.val_file, args.test_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(args.resume_checkpoint)

    if not args.no_validate_counts:
        validate_count(args.train_file, args.expected_train_count, "train")
        validate_count(args.val_file, args.expected_val_count, "validation")
        if not args.skip_eval:
            validate_count(args.test_file, args.expected_test_count, "test")

    args.run_name = args.wandb_name or (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{experiment_label(args)}"
    )
    if args.output_dir is None:
        args.output_dir = args.output_root / args.force_mode / args.run_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output_dir / "checkpoints"
    if args.log_dir is None:
        args.log_dir = args.output_dir / "logs"
    if args.energy_reference is None:
        args.energy_reference = args.reference_root / f"{reference_label(args)}.json"
    return args


if __name__ == "__main__":
    main(parse_args())
