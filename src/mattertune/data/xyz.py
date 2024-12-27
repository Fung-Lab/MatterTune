from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import ase
from ase import Atoms
from ase.io import read
import numpy as np
from torch.utils.data import Dataset
from typing_extensions import override
import copy

from ..registry import data_registry
from .base import DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class XYZDatasetConfig(DatasetConfigBase):
    type: Literal["xyz"] = "xyz"
    """Discriminator for the XYZ dataset."""

    src: str | Path
    """The path to the XYZ dataset."""
    
    down_sample: int | None = None
    """Down sample the dataset"""
    
    down_sample_refill: bool = False
    """Refill the dataset after down sampling to achieve the same length as the original dataset"""

    @override
    def create_dataset(self):
        return XYZDataset(self)


class XYZDataset(Dataset[ase.Atoms]):
    def __init__(self, config: XYZDatasetConfig):
        super().__init__()
        self.config = config

        atoms_list = read(str(self.config.src), index=":")
        assert isinstance(atoms_list, list), "Expected a list of Atoms objects"
        if self.config.down_sample is not None:
            ori_length = len(atoms_list)
            down_indices = np.random.choice(ori_length, self.config.down_sample, replace=False)
            if self.config.down_sample_refill:
                refilled_down_indices = []
                for _ in range((ori_length // self.config.down_sample)):
                    refilled_down_indices.extend(copy.deepcopy(down_indices))
                if len(refilled_down_indices) != ori_length:
                    res = np.random.choice(len(down_indices), ori_length - len(refilled_down_indices), replace=False)
                    refilled_down_indices.extend([down_indices[i] for i in res])
                new_atoms_list = [copy.deepcopy(atoms_list[i]) for i in refilled_down_indices]
                atoms_list = new_atoms_list
            else:
                new_atoms_list = [copy.deepcopy(atoms_list[i]) for i in down_indices]
                atoms_list = new_atoms_list
        self.atoms_list: list[Atoms] = atoms_list
        log.info(f"Loaded {len(self.atoms_list)} atoms from {self.config.src}")

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        return self.atoms_list[idx]

    def __len__(self) -> int:
        return len(self.atoms_list)
