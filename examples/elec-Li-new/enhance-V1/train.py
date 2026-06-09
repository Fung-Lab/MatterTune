from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rich
import torch
from ase import Atoms
from ase.io import read
from lightning.pytorch import LightningDataModule, Trainer
from rich.progress import track
from torch.utils.data import BatchSampler, DataLoader, Dataset

import mattertune.configs as MC
from mattertune.configs import WandbLoggerConfig
from mattertune.data import MatterTuneDataModule
from mattertune.finetune.base import _SkipBatchError
from mattertune.finetune.data_util import MapDatasetWrapper
from mattertune.finetune.loss import compute_loss
from mattertune.main import load_finetuned_checkpoint


EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1")
TEST_DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/electrolyte")
DEFAULT_OUTPUT_PREFIX = "Li_electrolyte_V1"
DEFAULT_TRAIN_FILE = DATA_ROOT / f"{DEFAULT_OUTPUT_PREFIX}_all.xyz"
DEFAULT_PAIR_TRAIN_FILE = DATA_ROOT / f"{DEFAULT_OUTPUT_PREFIX}_pairs.xyz"
DEFAULT_TEST_FILE = TEST_DATA_ROOT / "Li_system_test_with_del.xyz"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "local_runs" / "enhance-V1"
DEFAULT_INIT_CHECKPOINT: Path | None = None
MODEL_TYPES = ("mattersim", "orb", "uma")
FORCE_MODES = ("direct", "conservative")
TRAIN_WITH_DELTA_E = "train_with_delta_e"
TRAIN_WITHOUT_DELTA_E = "train_without_delta_e"
PAIR_ID_KEYS = ("delta_pair_id", "lambda_pair_id", "pair_id")
PAIR_ROLE_KEYS = ("delta_pair_role", "lambda_pair_role", "pair_role")


def normalize_model_type(raw: str) -> str:
    model_type = raw.strip().lower()
    aliases = {
        "mattersim-1m": "mattersim",
        "mattersim_1m": "mattersim",
        "mattersim1m": "mattersim",
        "mattersim-5m": "mattersim",
        "mattersim_5m": "mattersim",
        "mattersim5m": "mattersim",
    }
    model_type = aliases.get(model_type, model_type)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type {raw!r}; expected one of {MODEL_TYPES}.")
    return model_type


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


def model_label(model_type: str, model_name: str) -> str:
    safe_name = normalize_model_name(model_type, model_name).replace("/", "_")
    return f"{normalize_model_type(model_type)}-{safe_name}"


def experiment_label(args: argparse.Namespace) -> str:
    return f"{model_label(args.model_type, args.model_name)}-{normalize_force_mode(args.force_mode)}"


def safe_float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


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


def supervision_mode(args: argparse.Namespace) -> str:
    if args.e_loss_weight > 0.0 and args.f_loss_weight > 0.0:
        if args.delta_e_loss_weight > 0.0:
            return TRAIN_WITH_DELTA_E
        return TRAIN_WITHOUT_DELTA_E
    raise ValueError(
        "Supported enhance-V1 modes require --e_loss_weight > 0 and "
        "--f_loss_weight > 0. Use --delta_e_loss_weight <= 0 for energy/force "
        "training, or > 0 for energy/force/delta-E pair training."
    )


def validate_supervision_args(args: argparse.Namespace) -> None:
    args.training_mode = supervision_mode(args)
    if args.training_mode == TRAIN_WITH_DELTA_E:
        if args.batch_size < 2 or args.batch_size % 2 != 0:
            raise ValueError(
                "Delta-E pair training requires --batch_size >= 2 and an even batch size."
            )
    if args.model_type == "mattersim" and args.force_mode != "conservative":
        raise ValueError("MatterSim only supports conservative forces in MatterTune.")


def build_config(args: argparse.Namespace):
    hparams = MC.MatterTunerConfig.draft()

    args.model_type = normalize_model_type(args.model_type)
    args.model_name = normalize_model_name(args.model_type, args.model_name)
    args.force_mode = normalize_force_mode(args.force_mode)

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
            conservative=args.force_mode == "conservative",
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
    ckpt_name = f"{experiment_label(args)}-{args.training_mode}-best"
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
        f"{experiment_label(args)}-{args.training_mode}"
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


class AtomsListDataset(Dataset[Atoms]):
    def __init__(self, atoms_list: list[Atoms]):
        self.atoms_list = atoms_list

    def __len__(self) -> int:
        return len(self.atoms_list)

    def __getitem__(self, index: int) -> Atoms:
        return self.atoms_list[index]


class PairBatchSampler(BatchSampler):
    def __init__(
        self,
        pair_indices: np.ndarray,
        *,
        pairs_per_batch: int,
        shuffle: bool,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        super().__init__(sampler=[], batch_size=2 * pairs_per_batch, drop_last=False)
        self.pair_indices = np.asarray(pair_indices, dtype=np.int64)
        self.pairs_per_batch = pairs_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"Invalid rank {self.rank} for {self.num_replicas} replicas.")
        self.epoch = 0

    def _rank_pair_indices(self, pair_indices: np.ndarray) -> np.ndarray:
        if self.num_replicas == 1 or len(pair_indices) == 0:
            return pair_indices

        local_count = int(np.ceil(len(pair_indices) / self.num_replicas))
        total_size = local_count * self.num_replicas
        if total_size > len(pair_indices):
            padding = np.resize(pair_indices, total_size - len(pair_indices))
            pair_indices = np.concatenate([pair_indices, padding])
        return pair_indices[self.rank:total_size:self.num_replicas]

    def __iter__(self) -> Iterator[list[int]]:
        pair_indices = self.pair_indices.copy()
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(pair_indices)
            self.epoch += 1
        pair_indices = self._rank_pair_indices(pair_indices)

        batch: list[int] = []
        for pair_index in pair_indices:
            batch.extend([int(2 * pair_index), int(2 * pair_index + 1)])
            if len(batch) == 2 * self.pairs_per_batch:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        if len(self.pair_indices) == 0:
            return 0
        local_count = int(np.ceil(len(self.pair_indices) / self.num_replicas))
        return int(np.ceil(local_count / self.pairs_per_batch))


def info_int(info: dict[str, Any], keys: tuple[str, ...], *, description: str) -> int:
    for key in keys:
        if key in info:
            return int(info[key])
    raise KeyError(f"Missing {description}; expected one of {keys}.")


def pair_id(info: dict[str, Any]) -> int:
    return info_int(info, PAIR_ID_KEYS, description="pair id")


def pair_role(info: dict[str, Any]) -> int:
    return info_int(info, PAIR_ROLE_KEYS, description="pair role")


def parent_frame(info: dict[str, Any]) -> int:
    return info_int(
        info,
        (
            "frame",
            "pair_parent_frame",
            "pair_parent_index",
            "delta_pair_parent_frame",
            "deleted_parent_frame",
            "lambda_source_parent_index",
            "lambda_parent_index",
        ),
        description="parent frame/source index",
    )


class DeltaPairDataModule(LightningDataModule):
    def __init__(
        self,
        pair_file: Path,
        *,
        train_split: float,
        max_parent_frame: int | None,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle_seed: int,
    ):
        super().__init__()
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError("Delta-E pair training requires an even --batch_size >= 2.")
        self.pair_file = pair_file
        self.train_split = train_split
        self.max_parent_frame = max_parent_frame
        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_seed = shuffle_seed

    @property
    def lightning_module(self):
        if self.trainer is None or self.trainer.lightning_module is None:
            raise ValueError("No LightningModule is attached to this data module.")
        return self.trainer.lightning_module

    def setup(self, stage: str | None = None) -> None:
        atoms_list = read(self.pair_file, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        if len(atoms_list) % 2:
            raise ValueError(f"Pair file must contain an even number of structures: {self.pair_file}")

        for index in range(0, len(atoms_list), 2):
            left = atoms_list[index].info
            right = atoms_list[index + 1].info
            if pair_role(left) != 0 or pair_role(right) != 1:
                raise ValueError(
                    f"Expected without_del/with_del pair at structures {index}/{index + 1} "
                    f"in {self.pair_file}"
                )
            if pair_id(left) != pair_id(right):
                raise ValueError(f"Mismatched pair ids at structures {index}/{index + 1}.")

        original_n_pairs = len(atoms_list) // 2
        if self.max_parent_frame is not None:
            filtered_atoms: list[Atoms] = []
            for index in range(0, len(atoms_list), 2):
                frame = parent_frame(atoms_list[index].info)
                if frame < self.max_parent_frame:
                    filtered_atoms.extend([atoms_list[index], atoms_list[index + 1]])
            if not filtered_atoms:
                raise ValueError(
                    "No delta pairs remain after filtering parent frames "
                    f"< {self.max_parent_frame}."
                )
            atoms_list = filtered_atoms
            print(
                "filtered delta pairs by parent frame: "
                f"kept {len(atoms_list) // 2}/{original_n_pairs} pairs "
                f"(frame < {self.max_parent_frame})"
            )

        self.dataset = AtomsListDataset(atoms_list)
        n_pairs = len(atoms_list) // 2
        pair_indices = np.arange(n_pairs)
        rng = np.random.default_rng(self.shuffle_seed)
        rng.shuffle(pair_indices)
        train_len = int(self.train_split * n_pairs)
        self.train_pair_indices = pair_indices[:train_len]
        self.val_pair_indices = pair_indices[train_len:]

    def _mapped_dataset(self) -> MapDatasetWrapper[Atoms, Any]:
        module = self.lightning_module

        def map_fn(atoms: Atoms):
            data = module.atoms_to_data(atoms, has_labels=True)
            data = module.cpu_data_transform(data)
            delta_pair_id = torch.tensor([pair_id(atoms.info)], dtype=torch.long)
            delta_pair_role = torch.tensor([pair_role(atoms.info)], dtype=torch.long)
            data.delta_pair_id = delta_pair_id
            data.delta_pair_role = delta_pair_role
            if hasattr(data, "system_features") and isinstance(data.system_features, dict):
                data.system_features["delta_pair_id"] = delta_pair_id
                data.system_features["delta_pair_role"] = delta_pair_role
            if isinstance(data, dict):
                data["delta_pair_id"] = delta_pair_id
                data["delta_pair_role"] = delta_pair_role
            return data

        return MapDatasetWrapper(self.dataset, map_fn)

    def _distributed_context(self) -> tuple[int, int]:
        if self.trainer is None:
            return 1, 0
        return (
            max(1, int(getattr(self.trainer, "world_size", 1))),
            int(getattr(self.trainer, "global_rank", 0)),
        )

    def train_dataloader(self):
        num_replicas, rank = self._distributed_context()
        return DataLoader(
            self._mapped_dataset(),
            batch_sampler=PairBatchSampler(
                self.train_pair_indices,
                pairs_per_batch=self.pairs_per_batch,
                shuffle=True,
                seed=self.shuffle_seed,
                num_replicas=num_replicas,
                rank=rank,
            ),
            collate_fn=self.lightning_module.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        num_replicas, rank = self._distributed_context()
        return DataLoader(
            self._mapped_dataset(),
            batch_sampler=PairBatchSampler(
                self.val_pair_indices,
                pairs_per_batch=self.pairs_per_batch,
                shuffle=False,
                seed=self.shuffle_seed,
                num_replicas=num_replicas,
                rank=rank,
            ),
            collate_fn=self.lightning_module.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def _zero_loss(module: Any) -> torch.Tensor:
    return sum(parameter.sum() * 0.0 for parameter in module.parameters())


def delta_pair_indices(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(batch, "delta_pair_id") and hasattr(batch, "delta_pair_role"):
        pair_id_tensor = batch.delta_pair_id
        role_tensor = batch.delta_pair_role
    elif hasattr(batch, "system_features") and "delta_pair_id" in batch.system_features:
        pair_id_tensor = batch.system_features["delta_pair_id"]
        role_tensor = batch.system_features["delta_pair_role"]
    elif isinstance(batch, dict) and "delta_pair_id" in batch:
        pair_id_tensor = batch["delta_pair_id"]
        role_tensor = batch["delta_pair_role"]
    else:
        raise AttributeError("Batch does not contain delta_pair_id/delta_pair_role metadata.")

    pair_ids = pair_id_tensor.reshape(-1).detach().cpu().tolist()
    roles = role_tensor.reshape(-1).detach().cpu().tolist()
    by_pair: dict[int, dict[int, int]] = {}
    for graph_index, (pair_id_value, role) in enumerate(zip(pair_ids, roles, strict=True)):
        by_pair.setdefault(int(pair_id_value), {})[int(role)] = graph_index

    normal_indices: list[int] = []
    deleted_indices: list[int] = []
    for members in by_pair.values():
        if 0 in members and 1 in members:
            normal_indices.append(members[0])
            deleted_indices.append(members[1])

    device = pair_id_tensor.device
    return (
        torch.tensor(normal_indices, dtype=torch.long, device=device),
        torch.tensor(deleted_indices, dtype=torch.long, device=device),
    )


def weighted_grad_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    if not loss.requires_grad:
        return torch.zeros((), device=loss.device, dtype=loss.dtype)
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = torch.zeros((), device=loss.device, dtype=loss.dtype)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().pow(2).sum()
    return torch.sqrt(total)


def should_log_grad_norms(module: Any, mode: str) -> bool:
    if mode != "train" or not bool(getattr(module, "_enhance_log_loss_grad_norms", False)):
        return False
    interval = max(1, int(getattr(module, "_enhance_grad_norm_log_every_n_steps", 50)))
    trainer = getattr(module, "trainer", None)
    global_step = int(getattr(trainer, "global_step", 0))
    return global_step % interval == 0


def log_component_grad_norms(
    module: Any,
    *,
    mode: str,
    component_losses: dict[str, torch.Tensor],
    sync_dist: bool,
    batch_size: int | None,
) -> None:
    if not should_log_grad_norms(module, mode):
        return
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        return
    for name, loss in component_losses.items():
        norm = weighted_grad_norm(loss, parameters)
        module.log(
            f"{mode}/loss_grad_norm/{name}",
            norm,
            on_step=True,
            on_epoch=False,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )


def attach_enhanced_loss(
    module: Any,
    *,
    delta_e_loss_weight: float,
    log_loss_grad_norms: bool,
    grad_norm_log_every_n_steps: int,
) -> None:
    def _common_step_enhanced(
        self: Any,
        batch: Any,
        mode: str,
        metrics: Any | None,
        log: bool = True,
    ):
        sync_dist = bool(getattr(getattr(self, "trainer", None), "world_size", 1) > 1)
        labels = self.batch_to_labels(batch)
        energy_label = labels.get("energy")
        log_batch_size = (
            int(energy_label.shape[0])
            if isinstance(energy_label, torch.Tensor) and energy_label.ndim > 0
            else None
        )
        try:
            output = self(batch, mode=mode)
        except _SkipBatchError:
            return {"predicted_properties": {}}, _zero_loss(self)

        predictions = output["predicted_properties"]
        normalization_ctx = None
        if len(self.normalizers) > 0:
            normalization_ctx = self.create_normalization_context_from_batch(batch)
            predictions, labels = self.normalize(predictions, labels, normalization_ctx)

        for key, value in labels.items():
            labels[key] = value.contiguous()

        losses: list[torch.Tensor] = []
        component_losses: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            loss = compute_loss(prop.loss, predictions[prop.name], labels[prop.name])
            weighted_loss = loss * prop.loss_coefficient
            losses.append(weighted_loss)
            component_losses[prop.name] = weighted_loss
            if log:
                self.log(
                    f"{mode}/{prop.name}_loss",
                    weighted_loss,
                    sync_dist=sync_dist,
                    batch_size=log_batch_size,
                )

        if normalization_ctx is not None:
            denorm_predictions, denorm_labels = self.denormalize(
                predictions,
                labels,
                normalization_ctx,
            )
        else:
            denorm_predictions, denorm_labels = predictions, labels

        if delta_e_loss_weight > 0.0:
            normal_idx, deleted_idx = delta_pair_indices(batch)
            if len(normal_idx) > 0:
                pred_delta = (
                    denorm_predictions["energy"][deleted_idx]
                    - denorm_predictions["energy"][normal_idx]
                )
                label_delta = (
                    denorm_labels["energy"][deleted_idx]
                    - denorm_labels["energy"][normal_idx]
                )
                delta_loss = torch.mean((pred_delta - label_delta) ** 2)
                delta_mae = torch.mean(torch.abs(pred_delta - label_delta))
            else:
                delta_loss = denorm_predictions["energy"].sum() * 0.0
                delta_mae = torch.zeros((), device=delta_loss.device, dtype=delta_loss.dtype)

            weighted_delta_loss = delta_loss * delta_e_loss_weight
            losses.append(weighted_delta_loss)
            component_losses["delta_e"] = weighted_delta_loss
            if log:
                self.log(
                    f"{mode}/delta_e_loss",
                    weighted_delta_loss,
                    sync_dist=sync_dist,
                    batch_size=log_batch_size,
                )
                self.log(
                    f"{mode}/delta_e_mse_eV2",
                    delta_loss,
                    sync_dist=sync_dist,
                    batch_size=log_batch_size,
                )
                self.log(
                    f"{mode}/delta_e_mae_eV",
                    delta_mae,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=log_batch_size,
                )
                self.log(
                    f"{mode}/n_delta_e_pairs",
                    float(len(normal_idx)),
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=log_batch_size,
                )

        total_loss = sum(losses)
        if log:
            log_component_grad_norms(
                self,
                mode=mode,
                component_losses=component_losses,
                sync_dist=sync_dist,
                batch_size=log_batch_size,
            )
            self.log(
                f"{mode}/total_loss",
                total_loss,
                sync_dist=sync_dist,
                batch_size=log_batch_size,
            )

        if log and metrics is not None:
            self.log_dict(
                {
                    f"{mode}/{metric_name}": metric
                    for metric_name, metric in metrics(denorm_predictions, denorm_labels).items()
                },
                on_epoch=True,
                sync_dist=True,
                batch_size=log_batch_size,
            )

        return output, total_loss

    module._enhance_log_loss_grad_norms = log_loss_grad_norms
    module._enhance_grad_norm_log_every_n_steps = grad_norm_log_every_n_steps
    module._common_step = MethodType(_common_step_enhanced, module)


def initialize_from_checkpoint(model: Any, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint has no state_dict: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    print(
        "initialized model weights from "
        f"{checkpoint_path} "
        f"(epoch={checkpoint.get('epoch')}, global_step={checkpoint.get('global_step')})"
    )


def fit_enhance(args: argparse.Namespace) -> tuple[Any, Trainer]:
    config = build_config(args)
    config.model.ensure_dependencies()
    model = config.model.create_model()
    initialize_from_checkpoint(model, args.init_checkpoint)

    if args.training_mode == TRAIN_WITH_DELTA_E or args.log_loss_grad_norms:
        attach_enhanced_loss(
            model,
            delta_e_loss_weight=args.delta_e_loss_weight if args.training_mode == TRAIN_WITH_DELTA_E else 0.0,
            log_loss_grad_norms=args.log_loss_grad_norms,
            grad_norm_log_every_n_steps=args.grad_norm_log_every_n_steps,
        )

    if args.training_mode == TRAIN_WITH_DELTA_E:
        datamodule: LightningDataModule = DeltaPairDataModule(
            args.pair_train_file,
            train_split=args.train_split,
            max_parent_frame=args.max_parent_frame,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            shuffle_seed=args.shuffle_seed,
        )
    else:
        datamodule = MatterTuneDataModule(config.data)

    trainer_kwargs = config.trainer._to_lightning_kwargs()
    if model.requires_disabled_inference_mode():
        trainer_kwargs["inference_mode"] = False
    if args.training_mode == TRAIN_WITH_DELTA_E:
        trainer_kwargs["use_distributed_sampler"] = False
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule, ckpt_path=config.trainer.fit_ckpt_path())
    return model, trainer


def structure_group(atoms: Atoms) -> str:
    role = None
    for key in PAIR_ROLE_KEYS:
        if key in atoms.info:
            role = int(atoms.info[key])
            break
    if role == 0:
        return "without_del"
    if role == 1:
        return "with_del"
    if str(atoms.info.get("pair_role_name", "")).lower() in {"without_del", "with_del"}:
        return str(atoms.info["pair_role_name"]).lower()
    return "without_del" if "config_type" in atoms.info else "with_del"


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
    rich.print(f"enhance-V1 supervision mode: {args.training_mode}")
    _, trainer = fit_enhance(args)

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
    parser.add_argument("--model_type", choices=MODEL_TYPES, default="uma")
    parser.add_argument("--model_name", default="uma-s1.1")
    parser.add_argument(
        "--force_mode",
        choices=FORCE_MODES,
        default="conservative",
        help="Use a direct force head or conservative forces from the energy head.",
    )
    parser.add_argument(
        "--task_name",
        default="omol",
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
    parser.add_argument("--pair_train_file", type=Path, default=DEFAULT_PAIR_TRAIN_FILE)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--energy_reference", type=Path, required=True)
    parser.add_argument("--init_checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument(
        "--max_parent_frame",
        type=int,
        default=-1,
        help="Keep only delta pairs whose parent frame is below this value. Set <0 to disable.",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--e_loss_weight", type=float, default=200.0)
    parser.add_argument("--f_loss_weight", type=float, default=20.0)
    parser.add_argument(
        "--delta_e_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Set <=0 for train_without_delta_e. Set >0 with positive energy and "
            "force weights for train_with_delta_e."
        ),
    )
    parser.add_argument(
        "--log_loss_grad_norms",
        action="store_true",
        help=(
            "Log per-loss weighted gradient norms. This adds extra autograd.grad "
            "calls, so it is disabled by default."
        ),
    )
    parser.add_argument(
        "--grad_norm_log_every_n_steps",
        type=int,
        default=50,
        help="Training-step interval for --log_loss_grad_norms.",
    )
    parser.add_argument("--monitor", default="val/total_loss")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--gradient_clip_val", type=float, default=2.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--logger", choices=("wandb", "csv"), default="wandb")
    parser.add_argument("--wandb_project", default="MatterTune-Electrolyte-Li-enhance-V1")
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
    args.force_mode = normalize_force_mode(args.force_mode)
    args.devices = normalize_devices(args.devices)
    args.per_atom_energy_normalize = not args.no_per_atom_energy_normalize
    if args.max_parent_frame is not None and args.max_parent_frame < 0:
        args.max_parent_frame = None
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError(
            "--init_checkpoint only initializes model weights for a new run; "
            "--resume_checkpoint restores full training state. Use only one."
        )
    validate_supervision_args(args)

    mode_suffix = (
        f"{args.training_mode}-ew{safe_float_label(args.e_loss_weight)}-"
        f"fw{safe_float_label(args.f_loss_weight)}-dew{safe_float_label(args.delta_e_loss_weight)}"
    )
    run_name = args.wandb_name or (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{experiment_label(args)}-{mode_suffix}"
    )
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / args.training_mode / run_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(args.output_dir) / "checkpoints"
    if args.log_dir is None:
        args.log_dir = Path(args.output_dir) / "logs"

    required_paths = [args.train_file, args.test_file, args.energy_reference]
    if args.training_mode == TRAIN_WITH_DELTA_E:
        required_paths.append(args.pair_train_file)
    if args.init_checkpoint is not None:
        required_paths.append(args.init_checkpoint)
    if args.resume_checkpoint is not None:
        required_paths.append(args.resume_checkpoint)
    for required in required_paths:
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    return args


if __name__ == "__main__":
    main(parse_args())
