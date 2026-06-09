from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from lightning.pytorch import LightningDataModule, Trainer
from rich.progress import track
from torch.utils.data import BatchSampler, DataLoader, Dataset

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import mattertune.configs as MC
from mattertune.finetune.base import _SkipBatchError
from mattertune.finetune.data_util import MapDatasetWrapper
from mattertune.finetune.loss import compute_loss
from mattertune.loggers import WandbLoggerConfig

from train import (  # noqa: E402
    DATA_ROOT,
    DEFAULT_TRAIN_FILE,
    DEFAULT_TEST_FILE,
    MODEL_TYPES,
    build_config,
    evaluate_checkpoint,
    experiment_label,
    normalize_devices,
    normalize_force_mode,
    normalize_model_name,
    normalize_model_type,
)


DEFAULT_PAIR_TRAIN_FILE = DATA_ROOT / "Li_system_lambda_parent_del_pairs.xyz"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "local_runs" / "03-train-pair-100-uma"
DEFAULT_INIT_CHECKPOINT: Path | None = None
PAIR_ID_KEYS = ("delta_pair_id", "lambda_pair_id")
PAIR_ROLE_KEYS = ("delta_pair_role", "lambda_pair_role")


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
        ("frame", "delta_pair_parent_frame", "deleted_parent_frame", "lambda_source_parent_index"),
        description="parent frame/source index",
    )


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
        if batch_size < 2:
            raise ValueError("Delta-E pair training requires --batch_size >= 2.")
        self.pair_file = pair_file
        self.train_split = train_split
        self.max_parent_frame = max_parent_frame
        self.batch_size = batch_size
        self.pairs_per_batch = max(1, batch_size // 2)
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
                    f"Expected normal/deleted pair at structures {index}/{index + 1} in {self.pair_file}"
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
    for graph_index, (pair_id, role) in enumerate(zip(pair_ids, roles, strict=True)):
        by_pair.setdefault(int(pair_id), {})[int(role)] = graph_index

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


def _grad_l2_norm_stats(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[float, int]:
    if not loss.requires_grad:
        return 0.0, 0
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = 0.0
    nonnull = 0
    for grad in grads:
        if grad is None:
            continue
        nonnull += 1
        total += float(grad.detach().pow(2).sum().cpu())
    return float(total**0.5), nonnull


def _grad_l2_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    return _grad_l2_norm_stats(loss, parameters)[0]


def _write_grad_norm_row(output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "global_step",
        "epoch",
        "batch_idx",
        "energy_weight",
        "forces_weight",
        "delta_e_weight",
        "energy_loss",
        "forces_loss",
        "delta_e_loss",
        "total_loss",
        "energy_grad_norm",
        "forces_grad_norm",
        "delta_e_grad_norm",
        "total_grad_norm",
        "energy_loss_requires_grad",
        "forces_loss_requires_grad",
        "delta_e_loss_requires_grad",
        "energy_grad_nonnull_count",
        "forces_grad_nonnull_count",
        "delta_e_grad_nonnull_count",
        "total_grad_nonnull_count",
        "energy_raw_grad_norm",
        "forces_raw_grad_norm",
        "delta_e_raw_grad_norm",
        "energy_grad_share",
        "forces_grad_share",
        "delta_e_grad_share",
        "energy_to_forces_grad",
        "delta_e_to_forces_grad",
        "torch_grad_enabled",
        "module_training",
        "efs_head_training",
        "efs_inner_head_training",
    ]
    write_header = not output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def attach_delta_e_loss(module: Any, *, delta_e_loss_weight: float) -> None:
    def _common_step_with_delta(
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
        component_losses: dict[str, dict[str, torch.Tensor | float]] = {}
        for prop in self.hparams.properties:
            raw_loss = compute_loss(prop.loss, predictions[prop.name], labels[prop.name])
            weighted_loss = raw_loss * prop.loss_coefficient
            losses.append(weighted_loss)
            component_losses[prop.name] = {
                "raw_loss": raw_loss,
                "weighted_loss": weighted_loss,
                "weight": float(prop.loss_coefficient),
            }
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
            component_losses["delta_e"] = {
                "raw_loss": delta_loss,
                "weighted_loss": weighted_delta_loss,
                "weight": float(delta_e_loss_weight),
            }
        total_loss = sum(losses)
        if mode == "train":
            self._last_delta_component_losses = component_losses
        else:
            self._last_delta_component_losses = None

        if log:
            if delta_e_loss_weight > 0.0:
                self.log(f"{mode}/delta_e_loss", weighted_delta_loss, sync_dist=sync_dist, batch_size=log_batch_size)
                self.log(f"{mode}/delta_e_mse_eV2", delta_loss, sync_dist=sync_dist, batch_size=log_batch_size)
                self.log(f"{mode}/delta_e_mae_eV", delta_mae, on_epoch=True, sync_dist=True, batch_size=log_batch_size)
                self.log(f"{mode}/n_delta_e_pairs", float(len(normal_idx)), on_epoch=True, sync_dist=True, batch_size=log_batch_size)
            self.log(f"{mode}/total_loss", total_loss, sync_dist=sync_dist, batch_size=log_batch_size)

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

    module._common_step = MethodType(_common_step_with_delta, module)


def attach_gradient_norm_diagnostics(
    module: Any,
    *,
    output_path: Path,
    every_n_steps: int,
    max_records: int | None,
) -> None:
    if every_n_steps <= 0:
        raise ValueError("--grad_norm_every_n_steps must be positive when diagnostics are enabled.")

    original_training_step = module.training_step
    module._grad_norm_records_written = 0

    def _training_step_with_grad_norms(self: Any, batch: Any, batch_idx: int):
        loss = original_training_step(batch, batch_idx)
        trainer = getattr(self, "trainer", None)
        global_rank = int(getattr(trainer, "global_rank", 0)) if trainer is not None else 0
        global_step = int(getattr(self, "global_step", 0))
        should_record = (
            global_rank == 0
            and global_step % every_n_steps == 0
            and (max_records is None or self._grad_norm_records_written < max_records)
        )
        if not should_record:
            self._last_delta_component_losses = None
            return loss

        component_losses = getattr(self, "_last_delta_component_losses", None) or {}
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        names = ("energy", "forces", "delta_e")

        weighted_norms: dict[str, float] = {}
        grad_counts: dict[str, int] = {}
        loss_requires_grad: dict[str, bool] = {}
        raw_norms: dict[str, float] = {}
        weighted_losses: dict[str, float] = {}
        weights: dict[str, float] = {}
        for name in names:
            item = component_losses.get(name)
            if item is None:
                weighted_norms[name] = 0.0
                raw_norms[name] = 0.0
                weighted_losses[name] = 0.0
                weights[name] = 0.0
                grad_counts[name] = 0
                loss_requires_grad[name] = False
                continue
            weighted_loss = item["weighted_loss"]
            weight = float(item["weight"])
            weighted_norm, grad_count = _grad_l2_norm_stats(weighted_loss, parameters)
            weighted_norms[name] = weighted_norm
            grad_counts[name] = grad_count
            loss_requires_grad[name] = bool(weighted_loss.requires_grad)
            raw_norms[name] = weighted_norm / abs(weight) if weight != 0.0 else 0.0
            weighted_losses[name] = float(weighted_loss.detach().cpu())
            weights[name] = weight

        total_grad_norm, total_grad_count = _grad_l2_norm_stats(loss, parameters)
        norm_sum = sum(weighted_norms.values())
        forces_norm = weighted_norms["forces"]
        output_heads = getattr(self, "output_heads", None)
        efs_head = output_heads["efs"] if output_heads is not None and "efs" in output_heads else None
        efs_inner_head = getattr(efs_head, "head", efs_head) if efs_head is not None else None
        row = {
            "global_step": global_step,
            "epoch": int(getattr(self, "current_epoch", 0)),
            "batch_idx": batch_idx,
            "energy_weight": weights["energy"],
            "forces_weight": weights["forces"],
            "delta_e_weight": weights["delta_e"],
            "energy_loss": weighted_losses["energy"],
            "forces_loss": weighted_losses["forces"],
            "delta_e_loss": weighted_losses["delta_e"],
            "total_loss": float(loss.detach().cpu()),
            "energy_grad_norm": weighted_norms["energy"],
            "forces_grad_norm": weighted_norms["forces"],
            "delta_e_grad_norm": weighted_norms["delta_e"],
            "total_grad_norm": total_grad_norm,
            "energy_loss_requires_grad": int(loss_requires_grad["energy"]),
            "forces_loss_requires_grad": int(loss_requires_grad["forces"]),
            "delta_e_loss_requires_grad": int(loss_requires_grad["delta_e"]),
            "energy_grad_nonnull_count": grad_counts["energy"],
            "forces_grad_nonnull_count": grad_counts["forces"],
            "delta_e_grad_nonnull_count": grad_counts["delta_e"],
            "total_grad_nonnull_count": total_grad_count,
            "energy_raw_grad_norm": raw_norms["energy"],
            "forces_raw_grad_norm": raw_norms["forces"],
            "delta_e_raw_grad_norm": raw_norms["delta_e"],
            "energy_grad_share": weighted_norms["energy"] / norm_sum if norm_sum else 0.0,
            "forces_grad_share": weighted_norms["forces"] / norm_sum if norm_sum else 0.0,
            "delta_e_grad_share": weighted_norms["delta_e"] / norm_sum if norm_sum else 0.0,
            "energy_to_forces_grad": weighted_norms["energy"] / forces_norm if forces_norm else "",
            "delta_e_to_forces_grad": weighted_norms["delta_e"] / forces_norm if forces_norm else "",
            "torch_grad_enabled": int(torch.is_grad_enabled()),
            "module_training": int(bool(self.training)),
            "efs_head_training": int(bool(getattr(efs_head, "training", False))),
            "efs_inner_head_training": int(bool(getattr(efs_inner_head, "training", False))),
        }
        _write_grad_norm_row(output_path, row)
        self._grad_norm_records_written += 1
        self._last_delta_component_losses = None
        return loss

    module.training_step = MethodType(_training_step_with_grad_norms, module)


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


def fit_with_delta_pairs(args: argparse.Namespace) -> tuple[Any, Trainer]:
    config = build_config(args)
    config.model.ensure_dependencies()
    model = config.model.create_model()
    initialize_from_checkpoint(model, args.init_checkpoint)
    attach_delta_e_loss(model, delta_e_loss_weight=args.delta_e_loss_weight)
    if args.grad_norm_output is not None:
        attach_gradient_norm_diagnostics(
            model,
            output_path=args.grad_norm_output,
            every_n_steps=args.grad_norm_every_n_steps,
            max_records=args.grad_norm_max_records,
        )

    datamodule = DeltaPairDataModule(
        args.pair_train_file,
        train_split=args.train_split,
        max_parent_frame=args.max_parent_frame,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle_seed=args.shuffle_seed,
    )

    trainer_kwargs = config.trainer._to_lightning_kwargs()
    if model.requires_disabled_inference_mode():
        trainer_kwargs["inference_mode"] = False
    trainer_kwargs["use_distributed_sampler"] = False
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule, ckpt_path=config.trainer.fit_ckpt_path())
    return model, trainer


def main(args: argparse.Namespace) -> None:
    _, trainer = fit_with_delta_pairs(args)
    if args.skip_eval:
        print("skip_eval set; skipping test-set evaluation.")
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
        choices=("direct", "conservative"),
        default="conservative",
        help="Use a direct force head or conservative forces from the energy head.",
    )
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
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument(
        "--max_parent_frame",
        type=int,
        default=-1,
        help="Keep only delta pairs whose normal parent frame is below this value. Set <0 to disable.",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--e_loss_weight", type=float, default=200.0)
    parser.add_argument("--f_loss_weight", type=float, default=20.0)
    parser.add_argument("--delta_e_loss_weight", type=float, default=0.5)
    parser.add_argument("--monitor", default="val/total_loss")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--gradient_clip_val", type=float, default=2.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--logger", choices=("wandb", "csv"), default="wandb")
    parser.add_argument("--wandb_project", default="MatterTune-Electrolyte-Li-pair-100-UMA")
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
        "--grad_norm_output",
        type=Path,
        default=None,
        help="Optional CSV path for per-loss gradient norm diagnostics during training.",
    )
    parser.add_argument(
        "--grad_norm_every_n_steps",
        type=int,
        default=10,
        help="Record gradient norms every N training steps when --grad_norm_output is set.",
    )
    parser.add_argument(
        "--grad_norm_max_records",
        type=int,
        default=None,
        help="Maximum number of gradient norm records to write. Default records throughout training.",
    )
    parser.add_argument(
        "--reset_output_heads",
        action="store_true",
        help=(
            "Reset output heads before finetuning. Default is false so the "
            "energy reference remains aligned with the raw finetuning head."
        ),
    )
    parser.add_argument("--no_per_atom_energy_normalize", action="store_true")
    args = parser.parse_args()
    args.model_type = normalize_model_type(args.model_type)
    args.model_name = normalize_model_name(args.model_type, args.model_name)
    args.force_mode = normalize_force_mode(args.force_mode)
    if args.model_type != "mattersim" and args.init_checkpoint == DEFAULT_INIT_CHECKPOINT:
        args.init_checkpoint = None
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError(
            "--init_checkpoint only initializes model weights for a new run; "
            "--resume_checkpoint restores full training state. Use only one."
        )
    args.devices = normalize_devices(args.devices)
    args.per_atom_energy_normalize = not args.no_per_atom_energy_normalize
    if args.max_parent_frame is not None and args.max_parent_frame < 0:
        args.max_parent_frame = None

    run_name = args.wandb_name or (
        f"{__import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{experiment_label(args)}-pair100-fw20-de05"
    )
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / run_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(args.output_dir) / "checkpoints"
    if args.log_dir is None:
        args.log_dir = Path(args.output_dir) / "logs"

    required_paths = [args.train_file, args.pair_train_file, args.test_file, args.energy_reference]
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
