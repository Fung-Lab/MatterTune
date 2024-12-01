from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from torch.utils.data import Dataset
from typing_extensions import override

from ..registry import data_registry
from .base import DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class JSONDatasetConfig(DatasetConfigBase):
    type: Literal["json"] = "json"
    """Discriminator for the JSON dataset."""

    src: str | Path
    """The path to the JSON dataset."""

    tasks: dict[str, str]
    """Attributes in the JSON file that correspond to the tasks to be predicted."""

    @override
    def create_dataset(self):
        return JSONDataset(self)


class JSONDataset(Dataset[Atoms]):
    def __init__(self, config: JSONDatasetConfig):
        super().__init__()
        self.config = config

        with open(str(self.config.src), "r") as f:
            raw_data = json.load(f)

        self.atoms_list = []
        for entry in raw_data:
            atoms = Atoms(
                numbers=np.array(entry["atomic_numbers"]),
                positions=np.array(entry["positions"]),
                cell=np.array(entry["cell"]),
                pbc=True,
            )

            energy, forces, stress = None, None, None
            if "energy" in self.config.tasks:
                energy = torch.tensor(entry[self.config.tasks["energy"]])
            if "forces" in self.config.tasks:
                forces = torch.tensor(entry[self.config.tasks["forces"]])
            if "stress" in self.config.tasks:
                stress = torch.tensor(entry[self.config.tasks["stress"]])
                # ASE requires stress to be of shape (3, 3) or (6,)
                # Some datasets store stress with shape (1, 3, 3)
                if stress.ndim == 3:
                    stress = stress.squeeze(0)

            single_point_calc = SinglePointCalculator(
                atoms, energy=energy, forces=forces, stress=stress
            )

            atoms.calc = single_point_calc
            self.atoms_list.append(atoms)

        log.info(f"Loaded {len(self.atoms_list)} structures from {self.config.src}")

    @override
    def __getitem__(self, idx: int) -> Atoms:
        return self.atoms_list[idx]

    def __len__(self) -> int:
        return len(self.atoms_list)
