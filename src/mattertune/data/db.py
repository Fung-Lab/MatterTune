from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import ase
from ase import Atoms
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from typing_extensions import override

from ..registry import data_registry
from .base import DatasetBase, DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class DBDatasetConfig(DatasetConfigBase):
    """Configuration for a dataset stored in an ASE database."""

    type: Literal["db"] = "db"
    """Discriminator for the DB dataset."""

    src: str | Path
    """Path to the ASE database file."""

    energy_key: str | None = None
    """Key for the energy label in the database."""

    forces_key: str | None = None
    """Key for the force label in the database."""

    stress_key: str | None = None
    """Key for the stress label in the database."""

    @override
    @classmethod
    def dataset_cls(cls):
        return DBDataset


class DBDataset(DatasetBase[DBDatasetConfig]):
    def __init__(self, config: DBDatasetConfig):
        super().__init__(config)
        db = connect(config.src)
        self.atoms_list = []
        for row in db.select():
            atoms = self._load_atoms_from_row(row)
            self.atoms_list.append(atoms)

    def _load_atoms_from_row(self, row):
        atoms = row.toatoms()
        labels = dict(row.data)
        unrecognized_labels = {}
        if self.config.energy_key:
            labels["energy"] = labels.pop(self.config.energy_key)
        if self.config.forces_key:
            labels["forces"] = labels.pop(self.config.forces_key)
        if self.config.stress_key:
            labels["stress"] = labels.pop(self.config.stress_key)
        for key in list(labels.keys()):
            if key not in all_properties:
                unrecognized_labels[key] = labels.pop(key)
        calc = SinglePointCalculator(atoms, **labels)
        atoms.calc = calc
        atoms.info = unrecognized_labels
        return atoms

    @override
    def __getitem__(self, idx):
        return self.atoms_list[idx]

    def __len__(self):
        return len(self.atoms_list)
