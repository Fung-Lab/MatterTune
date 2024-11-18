from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import ase
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
from torch.utils.data.dataloader import _worker_init_fn_t
from typing_extensions import TypedDict, Unpack

from .data_util import IterableDatasetWrapper, MapDatasetWrapper

if TYPE_CHECKING:
    from .base import FinetuneModuleBase, TBatch, TData, TFinetuneModuleConfig


class DataLoaderKwargs(TypedDict, total=False):
    """Keyword arguments for creating a DataLoader.

    Args:
        batch_size: How many samples per batch to load (default: 1).
        shuffle: Set to True to have the data reshuffled at every epoch (default: False).
        sampler: Defines the strategy to draw samples from the dataset. Can be any Iterable with __len__
            implemented. If specified, shuffle must not be specified.
        batch_sampler: Like sampler, but returns a batch of indices at a time. Mutually exclusive with
            batch_size, shuffle, sampler, and drop_last.
        num_workers: How many subprocesses to use for data loading. 0 means that the data will be loaded
            in the main process (default: 0).
        pin_memory: If True, the data loader will copy Tensors into device/CUDA pinned memory before
            returning them.
        drop_last: Set to True to drop the last incomplete batch, if the dataset size is not divisible by
            the batch size (default: False).
        timeout: If positive, the timeout value for collecting a batch from workers. Should always be
            non-negative (default: 0).
        worker_init_fn: If not None, this will be called on each worker subprocess with the worker id
            as input, after seeding and before data loading.
        multiprocessing_context: If None, the default multiprocessing context of your operating system
            will be used.
        generator: If not None, this RNG will be used by RandomSampler to generate random indexes and
            multiprocessing to generate base_seed for workers.
        prefetch_factor: Number of batches loaded in advance by each worker.
        persistent_workers: If True, the data loader will not shut down the worker processes after a
            dataset has been consumed once.
        pin_memory_device: The device to pin_memory to if pin_memory is True.
    """

    batch_size: int | None
    shuffle: bool | None
    sampler: Sampler | Iterable | None
    batch_sampler: Sampler[list[int]] | Iterable[list[int]] | None
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    worker_init_fn: _worker_init_fn_t | None
    multiprocessing_context: Any  # type: ignore
    generator: Any  # type: ignore
    prefetch_factor: int | None
    persistent_workers: bool
    pin_memory_device: str


def create_dataloader(
    dataset: Dataset[ase.Atoms],
    has_labels: bool,
    *,
    lightning_module: FinetuneModuleBase[TData, TBatch, TFinetuneModuleConfig],
    **kwargs: Unpack[DataLoaderKwargs],
):
    def map_fn(ase_data: ase.Atoms):
        data = lightning_module.atoms_to_data(ase_data, has_labels)
        data = lightning_module.cpu_data_transform(data)
        return data

    # Wrap the dataset with the CPU data transform
    dataset_mapped = (
        IterableDatasetWrapper(dataset, map_fn)
        if isinstance(dataset, IterableDataset)
        else MapDatasetWrapper(dataset, map_fn)
    )
    # Create the data loader with the model's collate function
    dl = DataLoader(dataset_mapped, collate_fn=lightning_module.collate_fn, **kwargs)
    return dl
