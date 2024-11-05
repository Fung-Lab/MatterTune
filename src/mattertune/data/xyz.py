from __future__ import annotations

from pathlib import Path
from typing import Literal

import ase
from ase import Atoms
from ase.io import read
from typing_extensions import override

from ..registry import data_registry
from .base import DatasetBase, DatasetConfigBase


@data_registry.register
class XYZDatasetConfig(DatasetConfigBase):
    type: Literal["xyz"] = "xyz"
    """Discriminator for the XYZ dataset."""

    src: Path
    """The path to the XYZ dataset."""

    @override
    @classmethod
    def dataset_cls(cls):
        return XYZDataset


class XYZDataset(DatasetBase[XYZDatasetConfig]):
    def __init__(self, config: XYZDatasetConfig):
        super().__init__(config)

        self.atoms_list: list[Atoms] = read(str(self.config.src), index=":")

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        return self.atoms_list[idx]

    def __len__(self) -> int:
        return len(self.atoms_list)


