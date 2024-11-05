from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from ase import Atoms
from matbench.bench import MatbenchBenchmark
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from typing_extensions import override

from ..registry import data_registry
from .base import DatasetBase, DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class MatbenchDatasetConfig(DatasetConfigBase):
    """Configuration for the Matbench dataset."""

    type: Literal["matbench"] = "matbench"
    """Discriminator for the Matbench dataset."""

    task: str | None = None
    """The name of the self.tasks to include in the dataset."""

    property_name: str | None = None
    """The name of the property for the self.task."""

    fold_idx: Literal[0, 1, 2, 3, 4] = 0
    """The index of the fold to be used in the dataset."""

    split: Literal["train", "validation", "test"] = "train"

    train_split_ratio: float = 0.9
    """The ratio of the training data to the total train-valid data."""

    @override
    @classmethod
    def dataset_cls(cls):
        return MatbenchDataset


class MatbenchDataset(DatasetBase[MatbenchDatasetConfig]):
    def __init__(self, config: MatbenchDatasetConfig):
        super().__init__(config)
        self._config = config
        self._initialize_benchmark()
        self._load_data()

    def _initialize_benchmark(self) -> None:
        """Initialize the Matbench benchmark and task."""
        if self._config.task is None:
            mb = MatbenchBenchmark(autoload=False)
            all_tasks = list(mb.metadata.keys())
            raise ValueError(f"Please specify a task from {all_tasks}")
        else:
            mb = MatbenchBenchmark(autoload=False, subset=[self._config.task])
            self._task = mb.tasks[0]
            self._task.load()

    def _load_data(self) -> None:
        """Load and process the dataset split."""
        fold = self._task.folds[self._config.fold_idx]

        if self._config.split == "test":
            inputs = self._task.get_test_data(fold, include_target=False)
            self._atoms_list = self._convert_structures_to_atoms(inputs)
            log.info(f"Loaded test split with {len(self._atoms_list)} samples")
            return

        inputs_data, outputs_data = self._task.get_train_and_val_data(fold)
        split_idx = int(len(inputs_data) * self._config.train_split_ratio)

        if self._config.split == "train":
            inputs = inputs_data[:split_idx]
            outputs = outputs_data[:split_idx]
        else:  # validation
            inputs = inputs_data[split_idx:]
            outputs = outputs_data[split_idx:]

        self._atoms_list = self._convert_structures_to_atoms(inputs, outputs)
        log.info(
            f"Loaded {self._config.split} split with {len(self._atoms_list)} samples "
            f"(fold {self._config.fold_idx})"
        )

    def _convert_structures_to_atoms(
        self,
        structures: list[Structure],
        property_values: list[float] | None = None,
    ) -> list[Atoms]:
        """Convert pymatgen structures to ASE atoms.

        Args:
            structures: List of pymatgen Structure objects.
            property_values: Optional list of property values to add to atoms.info.

        Returns:
            List of ASE Atoms objects.
        """
        adapter = AseAtomsAdaptor()
        atoms_list = []
        prop_name = (
            self._config.property_name
            if self._config.property_name is not None
            else self._config.task
        )
        for i, structure in enumerate(structures):
            atoms = adapter.get_atoms(structure)
            if property_values is not None:
                atoms.info[prop_name] = property_values[i]
            atoms_list.append(atoms)

        return atoms_list

    @override
    def __getitem__(self, idx: int) -> Atoms:
        """Get an item from the dataset by index."""
        return self._atoms_list[idx]

    def __len__(self) -> int:
        """Get the total number of items in the dataset."""
        return len(self._atoms_list)

    def get_test_data(self) -> list[Atoms]:
        """Load the test data for the current task and fold.

        Returns:
            List of ASE Atoms objects from the test set.
        """
        if self._config.split == "test":
            return self._atoms_list

        test_inputs = self._task.get_test_data(
            self._task.folds[self._config.fold_idx], include_target=False
        )
        return self._convert_structures_to_atoms(test_inputs)
