from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import ase
import numpy as np
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.db.core import Database
from ase.stress import full_3x3_to_voigt_6_stress
from torch.utils.data import Dataset
from typing_extensions import override

from ..registry import data_registry
from .base import DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class DBDatasetConfig(DatasetConfigBase):
    """Configuration for a dataset stored in an ASE database."""

    type: Literal["db"] = "db"
    """Discriminator for the DB dataset."""

    src: Database | str | Path
    """Path to the ASE database file or a database object."""

    energy_key: str | None = None
    """Key for the energy label in the database."""

    forces_key: str | None = None
    """Key for the force label in the database."""

    stress_key: str | None = None
    """Key for the stress label in the database."""

    preload: bool = True
    """Whether to load all the data at once or not."""

    @override
    def create_dataset(self):
        return DBDataset(self)


class DBDataset(Dataset[ase.Atoms]):
    def __init__(self, config: DBDatasetConfig):
        super().__init__()
        self.config = config
        if isinstance(config.src, Database):
            self.db = config.src
        else:
            self.db = connect(config.src)
        if self.config.preload:
            self.atoms_list = []
            for row in self.db.select():
                atoms = self._load_atoms_from_row(row)
                self.atoms_list.append(atoms)

    def _load_atoms_from_row(self, row):
        atoms = row.toatoms()
        labels = dict(row.data)
        unrecognized_labels = {}
        if self.config.energy_key:
            labels["energy"] = labels.pop(self.config.energy_key)
        if self.config.forces_key:
            labels["forces"] = np.array(labels.pop(self.config.forces_key))
        if self.config.stress_key:
            labels["stress"] = np.array(labels.pop(self.config.stress_key))
            if labels["stress"].shape == (3, 3):
                labels["stress"] = full_3x3_to_voigt_6_stress(labels["stress"])
            elif labels["stress"].shape != (6,):
                raise ValueError(
                    f"Stress has unexpected shape: {labels['stress'].shape}, expected (3, 3) or (6,)"
                )
        for key in list(labels.keys()):
            if key not in all_properties:
                unrecognized_labels[key] = labels.pop(key)
        calc = SinglePointCalculator(atoms, **labels)
        atoms.calc = calc
        atoms.info = unrecognized_labels
        return atoms

    @override
    def __getitem__(self, idx):
        if self.config.preload:
            return self.atoms_list[idx]
        else:
            row = self.db.get(idx=idx)
            return self._load_atoms_from_row(row)

    def __len__(self):
        return len(self.db)
