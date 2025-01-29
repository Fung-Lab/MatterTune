from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Literal

import ase
import numpy as np
from ase import Atoms
from torch.utils.data import Dataset
from typing_extensions import override

from ..registry import data_registry
from .base import DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class AtomsListDatasetConfig(DatasetConfigBase):
    type: Literal["atoms_list"] = "atoms_list"
    """Discriminator for the atoms_list dataset."""

    atoms_list: list[ase.Atoms]
    """The list of Atoms objects."""

    @override
    def create_dataset(self):
        return AtomsListDataset(self)


class AtomsListDataset(Dataset[ase.Atoms]):
    def __init__(self, config: AtomsListDatasetConfig):
        super().__init__()
        self.config = config

        atoms_list = self.config.atoms_list
        assert isinstance(atoms_list, list), "Expected a list of Atoms objects"
        shuffle_indices = np.random.permutation(len(atoms_list))
        self.atoms_list = [atoms_list[i] for i in shuffle_indices]

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        return self.atoms_list[idx]

    def __len__(self) -> int:
        return len(self.atoms_list)
