from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import ase
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress
from typing_extensions import override

from ..registry import data_registry
from .base import DatasetBase, DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class MPTrajDatasetConfig(DatasetConfigBase):
    """Configuration for a dataset stored in the Materials Project database."""

    type: Literal["mptraj"] = "mptraj"
    """Discriminator for the MPTraj dataset."""

    split: Literal["train", "val", "test"] = "train"
    """Split of the dataset to use."""

    min_num_atoms: int | None = None
    """Minimum number of atoms to be considered. Drops structures with fewer atoms."""

    max_num_atoms: int | None = None
    """Maximum number of atoms to be considered. Drops structures with more atoms."""

    elements: list[str] | None = None
    """
    List of elements to be considered. Drops structures with elements not in the list.
    Subsets are also allowed. For example, ["Li", "Na"] will keep structures with either Li or Na.
    """

    @classmethod
    @override
    def dataset_cls(cls):
        return MPTrajDataset


class MPTrajDataset(DatasetBase[MPTrajDatasetConfig]):
    def __init__(self, config: MPTrajDatasetConfig):
        super().__init__(config)
        import datasets

        dataset = datasets.load_dataset("nimashoghi/mptrj", split=self.config.split)
        assert isinstance(dataset, datasets.Dataset)
        dataset.set_format("numpy")
        self.atoms_list = []
        for entry in dataset:
            atoms = self._load_atoms_from_entry(dict(entry))
            if self._filter_atoms(atoms):
                self.atoms_list.append(atoms)

    def _load_atoms_from_entry(self, entry: dict) -> Atoms:
        atoms = Atoms(
            positions=entry["positions"],
            numbers=entry["numbers"],
            cell=entry["cell"],
            pbc=True,
        )
        labels = {
            "energy": entry["corrected_total_energy"].item(),
            "forces": entry["forces"],
            "stress": full_3x3_to_voigt_6_stress(entry["stress"]),
        }
        calc = SinglePointCalculator(atoms, **labels)
        atoms.calc = calc
        return atoms

    def _filter_atoms(self, atoms: Atoms) -> bool:
        if (
            self.config.min_num_atoms is not None
            and len(atoms) < self.config.min_num_atoms
        ):
            return False
        if (
            self.config.max_num_atoms is not None
            and len(atoms) > self.config.max_num_atoms
        ):
            return False
        if self.config.elements is not None:
            elements = set(atoms.get_chemical_symbols())
            if not set(self.config.elements) >= elements:
                return False
        return True

    @override
    def __getitem__(self, idx: int) -> Atoms:
        return self.atoms_list[idx]

    def __len__(self):
        return len(self.atoms_list)
