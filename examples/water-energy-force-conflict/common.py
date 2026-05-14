from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rich
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

import mattertune.configs as MC
from mattertune.configs import WandbLoggerConfig
from mattertune.data import MatterTuneDataModule
from mattertune.finetune.loss import compute_loss
from mattertune.main import load_finetuned_checkpoint, load_pretrained_model


EXAMPLE_ROOT = Path(__file__).resolve().parent
MATTERTUNE_ROOT = EXAMPLE_ROOT.parents[1]
DEFAULT_DATA_ROOT = MATTERTUNE_ROOT / "examples" / "water-thermodynamics" / "data"
DEFAULT_TRAIN_FILE = DEFAULT_DATA_ROOT / "train_water_1000_eVAng.xyz"
DEFAULT_VAL_FILE = DEFAULT_DATA_ROOT / "val_water_1000_eVAng.xyz"
DEFAULT_ENERGY_REFERENCE = DEFAULT_DATA_ROOT / "water_1000_eVAng-energy_reference.json"


def str_to_bool(value: str | bool | int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


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


def default_device(args: argparse.Namespace) -> str:
    if getattr(args, "eval_device", ""):
        requested = str(args.eval_device)
    elif getattr(args, "accelerator", "gpu") == "cpu":
        requested = "cpu"
    else:
        requested = f"cuda:{args.devices[0]}"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        rich.print(
            "[yellow]CUDA was requested for evaluation/preflight, but this "
            "Python environment reports torch.cuda.is_available() == False. "
            "Using CPU for evaluation/preflight.[/yellow]"
        )
        return "cpu"
    return requested


def json_sanitize(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_sanitize(payload), handle, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_sanitize(payload), sort_keys=True) + "\n")


def clone_atoms_with_results(atoms: Atoms) -> Atoms:
    cloned = atoms.copy()
    cloned.info = copy.deepcopy(atoms.info)
    if atoms.calc is None:
        return cloned

    results: dict[str, Any] = {}
    for key, value in atoms.calc.results.items():
        results[key] = copy.deepcopy(value)
    if results:
        cloned.calc = SinglePointCalculator(cloned, **results)
    return cloned


@dataclass(frozen=True)
class PreparedData:
    source_train_file: Path
    effective_train_file: Path
    unique_train_file: Path
    val_file: Path
    selected_indices: list[int]
    effective_indices: list[int]
    source_train_size: int
    unique_train_size: int
    effective_train_size: int


def prepare_train_data(
    *,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    train_down_sample: int | None,
    down_sample_refill: bool,
    seed: int,
) -> PreparedData:
    atoms_list: list[Atoms] = read(train_file, index=":")  # type: ignore[assignment]
    if not atoms_list:
        raise ValueError(f"No structures loaded from {train_file}")

    source_size = len(atoms_list)
    rng = np.random.default_rng(seed)
    if train_down_sample is None or train_down_sample <= 0:
        selected_indices = list(range(source_size))
    else:
        if train_down_sample > source_size:
            raise ValueError(
                f"train_down_sample={train_down_sample} exceeds source size {source_size}"
            )
        selected_indices = rng.choice(
            source_size, size=train_down_sample, replace=False
        ).tolist()

    effective_indices = list(selected_indices)
    if down_sample_refill and len(selected_indices) < source_size:
        effective_indices = []
        repeats = source_size // len(selected_indices)
        for _ in range(repeats):
            effective_indices.extend(selected_indices)
        remainder = source_size - len(effective_indices)
        if remainder:
            refill_pick = rng.choice(
                len(selected_indices), size=remainder, replace=False
            ).tolist()
            effective_indices.extend(selected_indices[i] for i in refill_pick)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    unique_train_file = data_dir / "train_unique.xyz"
    effective_train_file = data_dir / "train_effective.xyz"
    indices_file = data_dir / "selection_indices.json"

    unique_atoms = [clone_atoms_with_results(atoms_list[i]) for i in selected_indices]
    effective_atoms = [
        clone_atoms_with_results(atoms_list[i]) for i in effective_indices
    ]
    write(unique_train_file, unique_atoms, format="extxyz")
    write(effective_train_file, effective_atoms, format="extxyz")

    prepared = PreparedData(
        source_train_file=train_file,
        effective_train_file=effective_train_file,
        unique_train_file=unique_train_file,
        val_file=val_file,
        selected_indices=[int(i) for i in selected_indices],
        effective_indices=[int(i) for i in effective_indices],
        source_train_size=source_size,
        unique_train_size=len(selected_indices),
        effective_train_size=len(effective_indices),
    )
    write_json(indices_file, prepared.__dict__)
    return prepared


def configure_model(
    *,
    model_type: str,
    lr: float,
    weight_decay: float,
    e_loss_weight: float,
    f_loss_weight: float,
    conservative: bool,
    reset_output_heads: bool,
    monitor: str = "val/total_loss",
    disable_lr_scheduler: bool = False,
    lr_patience: int = 5,
):
    from mattertune.backbones.jmp.model import get_jmp_s_lr_decay

    normalized = model_type.lower()
    if normalized in {"mattersim-1m", "mattersim-v1.0.0-1m"}:
        model = MC.MatterSimBackboneConfig.draft()
        model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        model.pretrained_model = "MatterSim-v1.0.0-1M"
    elif normalized in {"mattersim-5m", "mattersim-v1.0.0-5m"}:
        model = MC.MatterSimBackboneConfig.draft()
        model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        model.pretrained_model = "MatterSim-v1.0.0-5M"
    elif normalized == "jmp-s":
        model = MC.JMPBackboneConfig.draft()
        model.graph_computer = MC.JMPGraphComputerConfig.draft()
        model.graph_computer.pbc = True
        model.pretrained_model = "jmp-s"
    elif "orb" in normalized:
        model = MC.ORBBackboneConfig.draft()
        model.pretrained_model = model_type
    elif normalized == "eqv2":
        model = MC.EqV2BackboneConfig.draft()
        model.checkpoint_path = Path(
            "/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/eqV2_31M_mp.pt"
        )
        model.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig.draft()
        model.atoms_to_graph.radius = 8.0
        model.atoms_to_graph.max_num_neighbors = 20
    elif "mace" in normalized:
        model = MC.MACEBackboneConfig.draft()
        model.pretrained_model = model_type
    elif "nequip" in normalized:
        model = MC.NequIPBackboneConfig.draft()
        model.pretrained_model = model_type
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.ignore_gpu_batch_transform_error = True
    model.freeze_backbone = False
    model.reset_output_heads = reset_output_heads
    model.optimizer = MC.AdamWConfig(
        lr=lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=weight_decay,
        per_parameter_hparams=(
            get_jmp_s_lr_decay(lr) if "jmp" in normalized else None
        ),
    )
    if disable_lr_scheduler:
        model.lr_scheduler = None
    else:
        model.lr_scheduler = MC.ReduceOnPlateauConfig(
            mode="min",
            monitor=monitor,
            factor=0.8,
            patience=lr_patience,
            min_lr=1.0e-8,
        )
    model.properties = [
        MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(),
            loss_coefficient=e_loss_weight,
        ),
        MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(),
            conservative=conservative,
            loss_coefficient=f_loss_weight,
        ),
    ]
    return model


def build_config(args: argparse.Namespace, prepared_data: PreparedData):
    hparams = MC.MatterTunerConfig.draft()
    hparams.model = configure_model(
        model_type=args.model_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        e_loss_weight=args.e_loss_weight,
        f_loss_weight=args.f_loss_weight,
        conservative=args.conservative,
        reset_output_heads=args.reset_output_heads,
        monitor=args.monitor,
        disable_lr_scheduler=args.disable_lr_scheduler,
        lr_patience=args.lr_patience,
    )

    hparams.data = MC.ManualSplitDataModuleConfig.draft()
    hparams.data.train = MC.XYZDatasetConfig.draft()
    hparams.data.train.src = str(prepared_data.effective_train_file)
    hparams.data.validation = MC.XYZDatasetConfig.draft()
    hparams.data.validation.src = str(prepared_data.val_file)
    hparams.data.batch_size = args.batch_size
    hparams.data.pin_memory = False
    hparams.data.num_workers = args.num_workers

    normalizers = [
        MC.PerAtomReferencingNormalizerConfig(
            per_atom_references=Path(args.energy_reference)
        )
    ]
    if args.per_atom_energy_normalize:
        normalizers.append(MC.PerAtomNormalizerConfig())
    hparams.model.normalizers = {"energy": normalizers}

    hparams.trainer = MC.TrainerConfig.draft()
    hparams.trainer.max_epochs = args.max_epochs
    hparams.trainer.accelerator = args.accelerator
    if args.accelerator == "cpu":
        hparams.trainer.devices = max(1, args.devices[0])
    else:
        hparams.trainer.devices = args.devices
    if args.accelerator != "cpu" and len(args.devices) > 1:
        hparams.trainer.strategy = "ddp"
    hparams.trainer.gradient_clip_algorithm = "norm"
    hparams.trainer.gradient_clip_val = args.gradient_clip_val
    hparams.trainer.precision = "32"
    hparams.trainer.checkpoint = None
    if args.patience > 0:
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=args.monitor,
            patience=args.patience,
            mode="min",
            min_delta=1.0e-5,
        )

    run_name = args.run_name
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "cli": json_sanitize(vars(args)),
        "prepared_data": json_sanitize(prepared_data.__dict__),
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

    additional_trainer_kwargs: dict[str, Any] = {"inference_mode": False}
    if args.limit_train_batches is not None:
        additional_trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        additional_trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    hparams.trainer.additional_trainer_kwargs = additional_trainer_kwargs
    return hparams.finalize(strict=False)


def run_training(config, callbacks: list[Callback]):
    config.model.ensure_dependencies()
    lightning_module = config.model.create_model()
    datamodule = MatterTuneDataModule(config.data)
    trainer_kwargs: dict[str, Any] = config.trainer._to_lightning_kwargs()

    all_callbacks = list(trainer_kwargs.pop("callbacks", []))
    all_callbacks.extend(callbacks)
    trainer_kwargs["callbacks"] = all_callbacks
    if lightning_module.requires_disabled_inference_mode():
        trainer_kwargs["inference_mode"] = False

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, datamodule)
    return lightning_module, trainer


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def resolve_pretrained_model(model_type: str) -> tuple[str, str]:
    normalized = model_type.lower()
    if normalized in {"mattersim-1m", "mattersim-v1.0.0-1m"}:
        return "mattersim", "MatterSim-v1.0.0-1M"
    if normalized in {"mattersim-5m", "mattersim-v1.0.0-5m"}:
        return "mattersim", "MatterSim-v1.0.0-5M"
    if "orb" in normalized:
        return "orb", model_type
    if "mace" in normalized:
        return "mace", model_type
    if "nequip" in normalized:
        return "nequip", model_type
    raise NotImplementedError(
        f"Direct pretrained preflight is not available for model_type={model_type!r}."
    )


def raw_mattertune_calculator(args: argparse.Namespace):
    model_config = configure_model(
        model_type=args.model_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        e_loss_weight=1.0,
        f_loss_weight=1.0,
        conservative=args.conservative,
        reset_output_heads=False,
        monitor=args.monitor,
        disable_lr_scheduler=True,
    )
    model_config.normalizers = {}
    model_config = model_config.finalize(strict=False)
    model_config.ensure_dependencies()
    module = model_config.create_model()
    module.eval()
    return module.ase_calculator(device=default_device(args))


def pretrained_calculator(args: argparse.Namespace):
    try:
        family, model_name = resolve_pretrained_model(args.model_type)
        model = load_pretrained_model(
            family,
            model_name,
            device=default_device(args),
        )
        return model.ase_calculator(), f"load_pretrained_model:{family}:{model_name}"
    except Exception as exc:
        rich.print(
            "[yellow]Direct pretrained preflight failed; falling back to a raw "
            f"MatterTune module without normalizers. Reason: {exc}[/yellow]"
        )
        return raw_mattertune_calculator(args), "raw_mattertune_no_normalizer"


def evaluate_calculator(
    atoms_list: list[Atoms],
    calc,
) -> dict[str, Any]:
    true_energy: list[float] = []
    pred_energy: list[float] = []
    true_forces: list[np.ndarray] = []
    pred_forces: list[np.ndarray] = []
    natoms: list[int] = []

    for atoms in atoms_list:
        true_energy.append(float(atoms.get_potential_energy()))
        true_forces.append(np.asarray(atoms.get_forces(), dtype=np.float64))
        atoms_for_pred = atoms.copy()
        atoms_for_pred.calc = calc
        pred_energy.append(float(atoms_for_pred.get_potential_energy()))
        pred_forces.append(np.asarray(atoms_for_pred.get_forces(), dtype=np.float64))
        natoms.append(len(atoms))

    return {
        "true_energy": np.asarray(true_energy, dtype=np.float64),
        "pred_energy": np.asarray(pred_energy, dtype=np.float64),
        "true_forces": true_forces,
        "pred_forces": pred_forces,
        "natoms": np.asarray(natoms, dtype=np.float64),
    }


def force_component_mae(true_forces: list[np.ndarray], pred_forces: list[np.ndarray]) -> float:
    diff = np.vstack(pred_forces) - np.vstack(true_forces)
    return float(np.mean(np.abs(diff)))


def energy_metrics(true_energy: np.ndarray, pred_energy: np.ndarray) -> dict[str, float]:
    err = pred_energy - true_energy
    return {
        "mae_eV": float(np.mean(np.abs(err))),
        "rmse_eV": float(np.sqrt(np.mean(err**2))),
        "bias_eV": float(np.mean(err)),
        "residual_std_eV": float(np.std(err)),
        "residual_p95_minus_p05_eV": float(np.percentile(err, 95) - np.percentile(err, 5)),
    }


def run_preflight(args: argparse.Namespace, prepared_data: PreparedData) -> dict[str, Any]:
    output_path = Path(args.output_dir) / "preflight_summary.json"
    train_atoms: list[Atoms] = read(prepared_data.unique_train_file, index=":")  # type: ignore[assignment]
    val_atoms: list[Atoms] = read(prepared_data.val_file, index=":")  # type: ignore[assignment]
    if args.max_preflight_structures is not None:
        train_atoms = train_atoms[: args.max_preflight_structures]
        val_atoms = val_atoms[: args.max_preflight_structures]

    calc, source = pretrained_calculator(args)
    train_eval = evaluate_calculator(train_atoms, calc)
    val_eval = evaluate_calculator(val_atoms, calc)

    train_true = train_eval["true_energy"]
    train_pred = train_eval["pred_energy"]
    val_true = val_eval["true_energy"]
    val_pred = val_eval["pred_energy"]

    offset = float(np.mean(train_true - train_pred))
    affine_matrix = np.column_stack([train_pred, np.ones_like(train_pred)])
    affine_scale, affine_intercept = np.linalg.lstsq(
        affine_matrix, train_true, rcond=None
    )[0]
    affine_scale = float(affine_scale)
    affine_intercept = float(affine_intercept)

    val_offset_pred = val_pred + offset
    val_affine_pred = affine_scale * val_pred + affine_intercept
    train_offset_pred = train_pred + offset
    train_affine_pred = affine_scale * train_pred + affine_intercept

    train_rank_true = rankdata(train_true)
    train_rank_pred = rankdata(train_pred)
    val_rank_true = rankdata(val_true)
    val_rank_pred = rankdata(val_pred)

    summary = {
        "preflight_source": source,
        "constant_offset_threshold_eV_per_structure": args.offset_only_threshold,
        "is_mostly_zero_point_disagreement": bool(
            energy_metrics(val_true, val_offset_pred)["mae_eV"]
            < args.offset_only_threshold
        ),
        "calibration": {
            "constant_offset_eV": offset,
            "affine_scale": affine_scale,
            "affine_intercept_eV": affine_intercept,
        },
        "train": {
            "n_structures": len(train_true),
            "raw_energy": energy_metrics(train_true, train_pred),
            "offset_energy": energy_metrics(train_true, train_offset_pred),
            "affine_energy": energy_metrics(train_true, train_affine_pred),
            "force_component_mae_eV_A": force_component_mae(
                train_eval["true_forces"], train_eval["pred_forces"]
            ),
            "energy_pearson": safe_corr(train_true, train_pred),
            "energy_spearman": safe_corr(train_rank_true, train_rank_pred),
        },
        "val": {
            "n_structures": len(val_true),
            "raw_energy": energy_metrics(val_true, val_pred),
            "offset_energy": energy_metrics(val_true, val_offset_pred),
            "affine_energy": energy_metrics(val_true, val_affine_pred),
            "force_component_mae_eV_A": force_component_mae(
                val_eval["true_forces"], val_eval["pred_forces"]
            ),
            "energy_pearson": safe_corr(val_true, val_pred),
            "energy_spearman": safe_corr(val_rank_true, val_rank_pred),
        },
    }
    write_json(output_path, summary)
    rich.print(f"Saved preflight summary to {output_path}")
    return summary


def property_losses(pl_module, batch) -> dict[str, torch.Tensor]:
    labels = pl_module.batch_to_labels(batch)
    output = pl_module(batch, mode="train")
    predictions = output["predicted_properties"]
    if len(pl_module.normalizers) > 0:
        ctx = pl_module.create_normalization_context_from_batch(batch)
        predictions, labels = pl_module.normalize(predictions, labels, ctx)
    for key, value in labels.items():
        labels[key] = value.contiguous()

    losses: dict[str, torch.Tensor] = {}
    for prop in pl_module.hparams.properties:
        losses[prop.name] = compute_loss(prop.loss, predictions[prop.name], labels[prop.name])
    return losses


def trainable_named_parameters(pl_module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in pl_module.trainable_parameters() if param.requires_grad]


def grad_norm(grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
    total = None
    for grad in grads:
        if grad is None:
            continue
        value = torch.sum(grad.detach() * grad.detach())
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return torch.sqrt(total)


def grad_dot(
    lhs: Iterable[torch.Tensor | None],
    rhs: Iterable[torch.Tensor | None],
) -> torch.Tensor:
    total = None
    for left, right in zip(lhs, rhs):
        if left is None or right is None:
            continue
        value = torch.sum(left.detach() * right.detach())
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


class GradientDiagnosticsCallback(Callback):
    def __init__(
        self,
        *,
        output_path: Path,
        probe_interval: int,
    ):
        super().__init__()
        self.output_path = output_path
        self.requested_probe_interval = probe_interval
        self.probe_interval = probe_interval
        self.last_probe_step = -1
        self.initial_params: dict[str, torch.Tensor] = {}

    def on_fit_start(self, trainer, pl_module) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")
        self.initial_params = {
            name: param.detach().cpu().clone()
            for name, param in trainable_named_parameters(pl_module)
        }

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self.requested_probe_interval <= 0:
            num_batches = trainer.num_training_batches
            if isinstance(num_batches, int) and num_batches > 0:
                self.probe_interval = num_batches
            else:
                self.probe_interval = 1

    def should_probe(self, trainer) -> bool:
        if not trainer.is_global_zero:
            return False
        step = int(trainer.global_step)
        if step == 0 and self.last_probe_step < 0:
            return True
        return (step - self.last_probe_step) >= max(1, self.probe_interval)

    def parameter_displacement(self, pl_module) -> float:
        total = 0.0
        for name, param in trainable_named_parameters(pl_module):
            initial = self.initial_params.get(name)
            if initial is None:
                continue
            diff = param.detach().cpu() - initial
            total += float(torch.sum(diff * diff))
        return math.sqrt(total)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        if not self.should_probe(trainer):
            return

        named_params = trainable_named_parameters(pl_module)
        params = [param for _, param in named_params]
        pl_module.zero_grad(set_to_none=True)

        losses = property_losses(pl_module, batch)
        energy_loss = losses["energy"]
        force_loss = losses["forces"]
        energy_weight = float(
            next(prop.loss_coefficient for prop in pl_module.hparams.properties if prop.name == "energy")
        )
        force_weight = float(
            next(prop.loss_coefficient for prop in pl_module.hparams.properties if prop.name == "forces")
        )

        energy_grads = torch.autograd.grad(
            energy_loss,
            params,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=False,
        )
        force_grads = torch.autograd.grad(
            force_loss,
            params,
            retain_graph=False,
            allow_unused=True,
            materialize_grads=False,
        )

        energy_norm = float(grad_norm(energy_grads).cpu())
        force_norm = float(grad_norm(force_grads).cpu())
        weighted_energy_norm = abs(energy_weight) * energy_norm
        weighted_force_norm = abs(force_weight) * force_norm
        dot_ef = float(grad_dot(energy_grads, force_grads).cpu())
        cos_ef = safe_ratio(dot_ef, energy_norm * force_norm)

        total_sq = 0.0
        total_dot_energy = 0.0
        total_dot_force = 0.0
        for e_grad, f_grad in zip(energy_grads, force_grads):
            if e_grad is None and f_grad is None:
                continue
            if e_grad is None:
                total_grad = force_weight * f_grad.detach()
                e_detached = None
                f_detached = f_grad.detach()
            elif f_grad is None:
                total_grad = energy_weight * e_grad.detach()
                e_detached = e_grad.detach()
                f_detached = None
            else:
                e_detached = e_grad.detach()
                f_detached = f_grad.detach()
                total_grad = energy_weight * e_detached + force_weight * f_detached
            total_sq += float(torch.sum(total_grad * total_grad).cpu())
            if e_detached is not None:
                total_dot_energy += float(torch.sum(total_grad * e_detached).cpu())
            if f_detached is not None:
                total_dot_force += float(torch.sum(total_grad * f_detached).cpu())

        total_norm = math.sqrt(total_sq)
        payload = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "batch_idx": int(batch_idx),
            "probe_interval": int(self.probe_interval),
            "energy_loss_unweighted": float(energy_loss.detach().cpu()),
            "force_loss_unweighted": float(force_loss.detach().cpu()),
            "energy_loss_weighted": float((energy_loss * energy_weight).detach().cpu()),
            "force_loss_weighted": float((force_loss * force_weight).detach().cpu()),
            "energy_grad_norm": energy_norm,
            "force_grad_norm": force_norm,
            "weighted_energy_grad_norm": weighted_energy_norm,
            "weighted_force_grad_norm": weighted_force_norm,
            "weighted_force_to_energy_grad_norm_ratio": safe_ratio(
                weighted_force_norm, weighted_energy_norm
            ),
            "cos_energy_force": cos_ef,
            "total_grad_norm": total_norm,
            "cos_total_energy": safe_ratio(total_dot_energy, total_norm * energy_norm),
            "cos_total_force": safe_ratio(total_dot_force, total_norm * force_norm),
            "parameter_displacement_norm": self.parameter_displacement(pl_module),
            "non_none_energy_grads": sum(grad is not None for grad in energy_grads),
            "non_none_force_grads": sum(grad is not None for grad in force_grads),
        }
        append_jsonl(self.output_path, payload)
        pl_module.log_dict(
            {
                "grad/cos_energy_force": torch.tensor(
                    payload["cos_energy_force"] if payload["cos_energy_force"] is not None else 0.0
                ),
                "grad/weighted_force_to_energy_norm_ratio": torch.tensor(
                    payload["weighted_force_to_energy_grad_norm_ratio"]
                    if payload["weighted_force_to_energy_grad_norm_ratio"] is not None
                    else 0.0
                ),
                "grad/total_norm": torch.tensor(payload["total_grad_norm"]),
                "grad/parameter_displacement_norm": torch.tensor(
                    payload["parameter_displacement_norm"]
                ),
            },
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )
        pl_module.zero_grad(set_to_none=True)
        self.last_probe_step = int(trainer.global_step)


class ForceWeightScheduleCallback(Callback):
    def __init__(
        self,
        *,
        energy_weight: float,
        target_force_weight: float,
        energy_only_epochs: int,
        force_ramp_epochs: int,
    ):
        super().__init__()
        self.energy_weight = energy_weight
        self.target_force_weight = target_force_weight
        self.energy_only_epochs = energy_only_epochs
        self.force_ramp_epochs = force_ramp_epochs

    def force_weight_for_epoch(self, epoch: int) -> float:
        if epoch < self.energy_only_epochs:
            return 0.0
        if self.force_ramp_epochs <= 0:
            return self.target_force_weight
        ramp_index = epoch - self.energy_only_epochs + 1
        if ramp_index <= self.force_ramp_epochs:
            fraction = ramp_index / self.force_ramp_epochs
            return self.target_force_weight * fraction
        return self.target_force_weight

    def apply_weights(self, trainer, pl_module) -> None:
        force_weight = self.force_weight_for_epoch(int(trainer.current_epoch))
        for prop in pl_module.hparams.properties:
            if prop.name == "energy":
                prop.loss_coefficient = self.energy_weight
            elif prop.name == "forces":
                prop.loss_coefficient = force_weight
        pl_module.log("schedule/energy_loss_weight", self.energy_weight, on_epoch=True)
        pl_module.log("schedule/force_loss_weight", force_weight, on_epoch=True)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.apply_weights(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.apply_weights(trainer, pl_module)


def make_checkpoint_callbacks(checkpoint_dir: Path, labels: list[str]) -> list[ModelCheckpoint]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks: list[ModelCheckpoint] = []
    if "best-energy" in labels:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="best-energy",
                monitor="val/energy_mae",
                mode="min",
                save_top_k=1,
            )
        )
    if "best-total" in labels:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="best-total",
                monitor="val/total_loss",
                mode="min",
                save_top_k=1,
            )
        )
    return callbacks


def summarize_checkpoint_callbacks(callbacks: list[Callback]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for callback in callbacks:
        if not isinstance(callback, ModelCheckpoint):
            continue
        filename = str(callback.filename)
        if callback.best_model_path:
            paths[filename] = callback.best_model_path
    return paths


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    val_file: Path,
    device: str,
    max_eval_structures: int | None,
) -> dict[str, Any]:
    model = load_finetuned_checkpoint(str(checkpoint_path))
    calc = model.ase_calculator(device=device)
    atoms_list: list[Atoms] = read(val_file, index=":")  # type: ignore[assignment]
    if max_eval_structures is not None:
        atoms_list = atoms_list[:max_eval_structures]

    evaluated = evaluate_calculator(atoms_list, calc)
    true_e = evaluated["true_energy"]
    pred_e = evaluated["pred_energy"]
    natoms = evaluated["natoms"]
    epa_err = pred_e / natoms - true_e / natoms
    f_diff = np.vstack(evaluated["pred_forces"]) - np.vstack(evaluated["true_forces"])
    return {
        "checkpoint_path": str(checkpoint_path),
        "n_structures": int(len(true_e)),
        "energy_mae_eV": float(np.mean(np.abs(pred_e - true_e))),
        "energy_rmse_eV": float(np.sqrt(np.mean((pred_e - true_e) ** 2))),
        "energy_bias_eV": float(np.mean(pred_e - true_e)),
        "energy_per_atom_mae_eV": float(np.mean(np.abs(epa_err))),
        "force_component_mae_eV_A": float(np.mean(np.abs(f_diff))),
        "force_component_rmse_eV_A": float(np.sqrt(np.mean(f_diff**2))),
    }


def evaluate_checkpoints(
    *,
    checkpoint_paths: dict[str, str | Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for label, raw_path in checkpoint_paths.items():
        path = Path(raw_path)
        if not path.is_file():
            continue
        metrics[label] = evaluate_checkpoint(
            checkpoint_path=path,
            val_file=Path(args.val_file),
            device=default_device(args),
            max_eval_structures=args.max_eval_structures,
        )
    return metrics


def summarize_gradient_diagnostics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "n_probes": 0}

    cosines: list[float] = []
    ratios: list[float] = []
    total_energy_cosines: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload.get("cos_energy_force"), (int, float)):
                cosines.append(float(payload["cos_energy_force"]))
            if isinstance(
                payload.get("weighted_force_to_energy_grad_norm_ratio"), (int, float)
            ):
                ratios.append(float(payload["weighted_force_to_energy_grad_norm_ratio"]))
            if isinstance(payload.get("cos_total_energy"), (int, float)):
                total_energy_cosines.append(float(payload["cos_total_energy"]))

    summary: dict[str, Any] = {"path": str(path), "n_probes": len(cosines)}
    if cosines:
        cos_arr = np.asarray(cosines, dtype=np.float64)
        summary.update(
            {
                "cos_energy_force_median": float(np.median(cos_arr)),
                "cos_energy_force_mean": float(np.mean(cos_arr)),
                "cos_energy_force_negative_fraction": float(np.mean(cos_arr < 0.0)),
            }
        )
    if ratios:
        ratio_arr = np.asarray(ratios, dtype=np.float64)
        summary.update(
            {
                "weighted_force_to_energy_grad_norm_ratio_median": float(
                    np.median(ratio_arr)
                ),
                "weighted_force_to_energy_grad_norm_ratio_mean": float(
                    np.mean(ratio_arr)
                ),
            }
        )
    if total_energy_cosines:
        total_arr = np.asarray(total_energy_cosines, dtype=np.float64)
        summary.update(
            {
                "cos_total_energy_median": float(np.median(total_arr)),
                "cos_total_energy_negative_fraction": float(
                    np.mean(total_arr < 0.0)
                ),
            }
        )
    return summary


def add_common_args(parser: argparse.ArgumentParser, *, experiment_name: str) -> None:
    parser.add_argument("--model_type", default="mattersim-1m")
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val_file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--energy_reference", type=Path, default=DEFAULT_ENERGY_REFERENCE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--train_down_sample", type=int, default=30)
    parser.add_argument("--down_sample_refill", type=str_to_bool, default=True)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--e_loss_weight", type=float, default=1.0)
    parser.add_argument("--f_loss_weight", type=float, default=1.0)
    parser.add_argument("--conservative", type=str_to_bool, default=True)
    parser.add_argument("--reset_output_heads", type=str_to_bool, default=True)
    parser.add_argument("--monitor", default="val/total_loss")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--disable_lr_scheduler", action="store_true")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--logger", choices=("csv", "wandb"), default="csv")
    parser.add_argument("--wandb_project", default="MatterTune-Water-Energy-Force-Conflict")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--eval_device", default="")
    parser.add_argument("--max_eval_structures", type=int, default=None)
    parser.add_argument("--max_preflight_structures", type=int, default=None)
    parser.add_argument("--offset_only_threshold", type=float, default=0.1)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--grad_probe_interval", type=int, default=0)
    parser.add_argument("--no_per_atom_energy_normalize", action="store_true")


def finalize_common_args(args: argparse.Namespace, *, experiment_name: str) -> argparse.Namespace:
    args.devices = normalize_devices(args.devices)
    args.per_atom_energy_normalize = not args.no_per_atom_energy_normalize

    run_name = (
        args.run_name
        or args.wandb_name
        or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.model_type}"
    )
    args.run_name = run_name
    if not args.wandb_name:
        args.wandb_name = run_name

    if args.output_dir is None:
        args.output_dir = EXAMPLE_ROOT / "runs" / experiment_name / run_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(args.output_dir) / "checkpoints"
    if args.log_dir is None:
        args.log_dir = Path(args.output_dir) / "logs"

    for required in (args.train_file, args.val_file, args.energy_reference):
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    return args


def print_run_header(args: argparse.Namespace, prepared_data: PreparedData, experiment: str) -> None:
    rich.print(f"Experiment       : {experiment}")
    rich.print(f"Output dir       : {args.output_dir}")
    rich.print(f"Model type       : {args.model_type}")
    rich.print(f"Train source     : {prepared_data.source_train_file}")
    rich.print(f"Train effective  : {prepared_data.effective_train_file}")
    rich.print(f"Val file         : {prepared_data.val_file}")
    rich.print(f"Unique/effective : {prepared_data.unique_train_size}/{prepared_data.effective_train_size}")
    rich.print(f"Devices          : {args.devices}")
    rich.print(f"Loss weights     : energy={args.e_loss_weight}, forces={args.f_loss_weight}")
