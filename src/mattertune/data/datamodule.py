from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sized
from typing import TYPE_CHECKING, Annotated, Any, Literal

import ase
import nshconfig as C
import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset
from typing_extensions import TypeAliasType, TypedDict, override

from ..registry import data_registry
from .base import DatasetConfig
from .util.split_dataset import SplitDataset

if TYPE_CHECKING:
    from ..finetune.loader import DataLoaderKwargs

log = logging.getLogger(__name__)


class DatasetMapping(TypedDict, total=False):
    train: Dataset[ase.Atoms]
    validation: Dataset[ase.Atoms]


class DataModuleBaseConfig(C.Config, ABC):
    batch_size: int
    """The batch size for the dataloaders."""

    num_workers: int | Literal["auto"] = "auto"
    """The number of workers for the dataloaders.

    This is the number of processes that generate batches in parallel.

    If set to "auto", the number of workers will be automatically
    set based on the number of available CPUs.

    Set to 0 to disable parallelism.
    """

    pin_memory: bool = True
    """Whether to pin memory in the dataloaders.

    This is useful for speeding up GPU data transfer.
    """

    def _num_workers_or_auto(self):
        if self.num_workers == "auto":
            import os

            if (cpu_count := os.cpu_count()) is not None:
                return cpu_count - 1
            else:
                return 1

        return self.num_workers

    def dataloader_kwargs(self) -> DataLoaderKwargs:
        return {
            "batch_size": self.batch_size,
            "num_workers": self._num_workers_or_auto(),
            "pin_memory": self.pin_memory,
        }

    @abstractmethod
    def dataset_configs(self) -> Iterable[DatasetConfig]: ...

    @abstractmethod
    def create_datasets(self) -> DatasetMapping: ...


@data_registry.rebuild_on_registers
class ManualSplitDataModuleConfig(DataModuleBaseConfig):
    train: DatasetConfig
    """The configuration for the training data."""

    validation: DatasetConfig | None = None
    """The configuration for the validation data."""

    @override
    def dataset_configs(self):
        yield self.train
        if self.validation is not None:
            yield self.validation

    @override
    def create_datasets(self):
        datasets: DatasetMapping = {}
        datasets["train"] = self.train.create_dataset()
        if (val := self.validation) is not None:
            datasets["validation"] = val.create_dataset()
        return datasets


@data_registry.rebuild_on_registers
class AutoSplitDataModuleConfig(DataModuleBaseConfig):
    dataset: DatasetConfig
    """The configuration for the dataset."""

    train_split: float
    """The proportion of the dataset to include in the training split."""

    validation_split: float | Literal["auto", "disable"] = "auto"
    """The proportion of the dataset to include in the validation split.

    If set to "auto", the validation split will be automatically determined as
    the complement of the training split, i.e. `validation_split = 1 - train_split`.

    If set to "disable", the validation split will be disabled.
    """

    shuffle: bool = True
    """Whether to shuffle the dataset before splitting."""

    shuffle_seed: int = 42
    """The seed to use for shuffling the dataset."""

    def _resolve_train_val_split(self):
        train_split = self.train_split
        match self.validation_split:
            case "auto":
                validation_split = 1.0 - train_split
            case "disable":
                validation_split = 0.0
            case _:
                validation_split = self.validation_split

        return train_split, validation_split

    @override
    def dataset_configs(self):
        yield self.dataset

    @override
    def create_datasets(self):
        # Create the full dataset.
        dataset = self.dataset.create_dataset()

        # If the validation split is disabled, return the full dataset.
        if self.validation_split == "disable":
            return DatasetMapping(train=dataset)

        if not isinstance(dataset, Sized):
            raise TypeError(
                f"The underlying dataset must be sized, but got {dataset!r}."
            )

        # Compute the indices for the training and validation splits.
        dataset_len = len(dataset)
        indices = np.arange(dataset_len)
        if self.shuffle:
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(indices)

        train_split, validation_split = self._resolve_train_val_split()
        train_len = int(train_split * dataset_len)
        validation_len = int(validation_split * dataset_len)

        # Get indices for each split
        train_indices = indices[:train_len]
        validation_indices = indices[train_len : train_len + validation_len]
        # Create the training and validation datasets.
        train_dataset = SplitDataset(dataset, train_indices)
        validation_dataset = SplitDataset(dataset, validation_indices)

        return DatasetMapping(train=train_dataset, validation=validation_dataset)


DataModuleConfig = TypeAliasType(
    "DataModuleConfig",
    Annotated[
        ManualSplitDataModuleConfig | AutoSplitDataModuleConfig,
        C.Field(description="The configuration for the data module."),
    ],
)


class MatterTuneDataModule(LightningDataModule):
    hparams: DataModuleConfig  # pyright: ignore[reportIncompatibleMethodOverride]
    hparams_initial: DataModuleConfig  # pyright: ignore[reportIncompatibleMethodOverride]

    @override
    def __init__(self, hparams: DataModuleConfig | Mapping[str, Any]):
        # Validate & resolve the configuration.
        if not isinstance(hparams, C.Config):
            hparams = C.TypeAdapter(DataModuleConfig).validate_python(hparams)

        super().__init__()

        # Save the configuration for Lightning.
        self.save_hyperparameters(hparams)

    @override
    def prepare_data(self) -> None:
        for config in self.hparams.dataset_configs():
            config.prepare_data()

    @override
    def setup(self, stage: str):
        super().setup(stage)

        self.datasets = self.hparams.create_datasets()

        # PyTorch Lightning checks for the *existence* of the
        # `train_dataloader`, `val_dataloader`, `test_dataloader`,
        # and `predict_dataloader` methods to determine which dataloaders
        # to create. That means that we cannot just return `None` from
        # these methods if the dataset is not available. We also cannot
        # raise an exception, because this will just crash the training
        # loop.
        # Instead, we will check, here, what datasets are available, and
        # remove the corresponding methods if the dataset is not available.
        METHOD_NAME_MAPPING = {
            "train": "train_dataloader",
            "validation": "val_dataloader",
        }
        for dataset_name, method_name in METHOD_NAME_MAPPING.items():
            if dataset_name not in self.datasets:
                setattr(self, method_name, None)

    @property
    def lightning_module(self):
        if (trainer := self.trainer) is None:
            raise ValueError("No trainer found.")

        if (lightning_module := trainer.lightning_module) is None:
            raise ValueError("No LightningModule found.")

        from ..finetune.base import FinetuneModuleBase

        if not isinstance(lightning_module, FinetuneModuleBase):
            raise ValueError("The LightningModule is not a FinetuneModuleBase.")

        return lightning_module

    @override
    def train_dataloader(self):
        if (dataset := self.datasets.get("train")) is None:
            raise ValueError("No training dataset found.")

        return self.lightning_module.create_dataloader(
            dataset,
            has_labels=True,
            **self.hparams.dataloader_kwargs(),
        )

    @override
    def val_dataloader(self):
        if (dataset := self.datasets.get("validation")) is None:
            raise ValueError(
                "No validation dataset found, but `val_dataloader` was called. "
                "This should not happen. Report this as a bug."
            )
        return self.lightning_module.create_dataloader(
            dataset,
            has_labels=True,
            **self.hparams.dataloader_kwargs(),
        )
