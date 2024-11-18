from __future__ import annotations

from collections.abc import Callable, Iterator, Sized
from typing import Any, Generic

from torch.utils.data import Dataset, IterableDataset
from typing_extensions import TypeVar, override

TDataIn = TypeVar("TDataIn", default=Any, infer_variance=True)
TDataOut = TypeVar("TDataOut", default=Any, infer_variance=True)


class MapDatasetWrapper(Dataset[TDataOut], Generic[TDataIn, TDataOut]):
    @override
    def __init__(
        self,
        dataset: Dataset[TDataIn],
        map_fn: Callable[[TDataIn], TDataOut],
    ):
        assert isinstance(
            dataset, Sized
        ), "The dataset must be sized. Otherwise, use _IterableDatasetWrapper."
        self.dataset = dataset
        self.map_fn = map_fn

    def __len__(self) -> int:
        return len(self.dataset)

    @override
    def __getitem__(self, idx: int) -> TDataOut:
        return self.map_fn(self.dataset[idx])


class IterableDatasetWrapper(IterableDataset[TDataOut], Generic[TDataIn, TDataOut]):
    @override
    def __init__(
        self,
        dataset: IterableDataset[TDataIn],
        map_fn: Callable[[TDataIn], TDataOut],
    ):
        self.dataset = dataset
        self.map_fn = map_fn

    @override
    def __iter__(self) -> Iterator[TDataOut]:
        for data in self.dataset:
            yield self.map_fn(data)
