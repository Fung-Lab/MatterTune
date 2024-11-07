from __future__ import annotations

from collections.abc import Sized
from typing import Generic, TypeVar

import ase
import numpy as np
from ase import Atoms
from torch.utils.data import Dataset
from typing_extensions import override

TDataset = TypeVar("TDataset", bound=Dataset[ase.Atoms], covariant=True)


class SplitDataset(Dataset[ase.Atoms], Generic[TDataset]):
    @override
    def __init__(self, dataset: TDataset, indices: np.ndarray):
        super().__init__()

        self.dataset = dataset
        self.indices = indices

        # Make sure the underlying dataset is a sized mappable dataset.
        if not isinstance(dataset, Sized):
            raise TypeError(
                f"The underlying dataset must be sized, but got {dataset!r}."
            )

        # Make sure the indices are valid.
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"The indices must be integers, but got {indices.dtype!r}.")

        if not (0 <= indices).all() and (indices < len(dataset)).all():
            raise ValueError(
                f"The indices must be in the range [0, {len(dataset)}), but got [{indices.min()}, {indices.max()}]."
            )

    def __len__(self) -> int:
        return len(self.indices)

    @override
    def __getitem__(self, index: int) -> Atoms:
        index = int(self.indices[index])
        return self.dataset[index]
