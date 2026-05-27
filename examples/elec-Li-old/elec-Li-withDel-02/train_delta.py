from __future__ import annotations

import argparse
import json
import os
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
    DEFAULT_TEST_FILE,
    build_config,
    evaluate_checkpoint,
    normalize_devices,
)


DEFAULT_PAIR_TRAIN_FILE = (
    DATA_ROOT / "local_runs" / "elec-Li-withDel-02" / "data" / "delta_pairs_train.xyz"
)
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "local_runs" / "elec-Li-withDel-02"


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
            if int(left.get("delta_pair_role", -1)) != 0 or int(right.get("delta_pair_role", -1)) != 1:
                raise ValueError(
                    f"Expected normal/deleted pair at structures {index}/{index + 1} in {self.pair_file}"
                )
            if int(left["delta_pair_id"]) != int(right["delta_pair_id"]):
                raise ValueError(f"Mismatched pair ids at structures {index}/{index + 1}.")

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
            data.delta_pair_id = torch.tensor([int(atoms.info["delta_pair_id"])], dtype=torch.long)
            data.delta_pair_role = torch.tensor([int(atoms.info["delta_pair_role"])], dtype=torch.long)
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
    pair_ids = batch.delta_pair_id.reshape(-1).detach().cpu().tolist()
    roles = batch.delta_pair_role.reshape(-1).detach().cpu().tolist()
    by_pair: dict[int, dict[int, int]] = {}
    for graph_index, (pair_id, role) in enumerate(zip(pair_ids, roles, strict=True)):
        by_pair.setdefault(int(pair_id), {})[int(role)] = graph_index

    normal_indices: list[int] = []
    deleted_indices: list[int] = []
    for members in by_pair.values():
        if 0 in members and 1 in members:
            normal_indices.append(members[0])
            deleted_indices.append(members[1])

    device = batch.delta_pair_id.device
    return (
        torch.tensor(normal_indices, dtype=torch.long, device=device),
        torch.tensor(deleted_indices, dtype=torch.long, device=device),
    )


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
        for prop in self.hparams.properties:
            loss = compute_loss(prop.loss, predictions[prop.name], labels[prop.name])
            loss = loss * prop.loss_coefficient
            losses.append(loss)
            if log:
                self.log(f"{mode}/{prop.name}_loss", loss, sync_dist=sync_dist)

        if normalization_ctx is not None:
            denorm_predictions, denorm_labels = self.denormalize(
                predictions,
                labels,
                normalization_ctx,
            )
        else:
            denorm_predictions, denorm_labels = predictions, labels

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
        total_loss = sum(losses)

        if log:
            self.log(f"{mode}/delta_e_loss", weighted_delta_loss, sync_dist=sync_dist)
            self.log(f"{mode}/delta_e_mse_eV2", delta_loss, sync_dist=sync_dist)
            self.log(f"{mode}/delta_e_mae_eV", delta_mae, on_epoch=True, sync_dist=True)
            self.log(f"{mode}/n_delta_e_pairs", float(len(normal_idx)), on_epoch=True, sync_dist=True)
            self.log(f"{mode}/total_loss", total_loss, sync_dist=sync_dist)

        if log and metrics is not None:
            self.log_dict(
                {
                    f"{mode}/{metric_name}": metric
                    for metric_name, metric in metrics(denorm_predictions, denorm_labels).items()
                },
                on_epoch=True,
                sync_dist=True,
            )

        return output, total_loss

    module._common_step = MethodType(_common_step_with_delta, module)


def fit_with_delta_pairs(args: argparse.Namespace) -> tuple[Any, Trainer]:
    config = build_config(args)
    config.model.ensure_dependencies()
    model = config.model.create_model()
    attach_delta_e_loss(model, delta_e_loss_weight=args.delta_e_loss_weight)

    datamodule = DeltaPairDataModule(
        args.pair_train_file,
        train_split=args.train_split,
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
    trainer.fit(model, datamodule)
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
    parser.add_argument("--model_name", default="MatterSim-v1.0.0-1M")
    parser.add_argument("--train_file", type=Path, default=DATA_ROOT / "Li_system_train_with_del.xyz")
    parser.add_argument("--pair_train_file", type=Path, default=DEFAULT_PAIR_TRAIN_FILE)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--energy_reference", type=Path, required=True)
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
    parser.add_argument("--delta_e_loss_weight", type=float, default=1.0)
    parser.add_argument("--monitor", default="val/total_loss")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--gradient_clip_val", type=float, default=2.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--logger", choices=("wandb", "csv"), default="wandb")
    parser.add_argument("--wandb_project", default="MatterTune-Electrolyte-Li-withDel-02")
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

    run_name = args.wandb_name or f"{__import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')}-mattersim-withDel-02"
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / run_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(args.output_dir) / "checkpoints"
    if args.log_dir is None:
        args.log_dir = Path(args.output_dir) / "logs"

    for required in (args.train_file, args.pair_train_file, args.test_file, args.energy_reference):
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    return args


if __name__ == "__main__":
    main(parse_args())
