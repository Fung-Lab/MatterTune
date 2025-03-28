from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import ase
from torch.utils.data import Dataset
from typing_extensions import override

from ..registry import data_registry
from ..util import optional_import_error_message
from .base import DatasetConfigBase

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

log = logging.getLogger(__name__)


@data_registry.register
class MatbenchDatasetConfig(DatasetConfigBase):
    """Configuration for the Matbench dataset."""

    type: Literal["matbench"] = "matbench"
    """Discriminator for the Matbench dataset."""

    task: str | None = None
    """The name of the self.tasks to include in the dataset."""

    property_name: str | None = None
    """Assign a property name for the self.task. Must match the property head in the model."""

    fold_idx: int = 0
    """The index of the fold to be used in the dataset."""

    @override
    def create_dataset(self):
        return MatbenchDataset(self)


class MatbenchDataset(Dataset[ase.Atoms]):
    def __init__(self, config: MatbenchDatasetConfig):
        super().__init__()
        self.config = config
        self._initialize_benchmark()
        self._load_data()

    def _initialize_benchmark(self) -> None:
        """Initialize the Matbench benchmark and task."""

        with optional_import_error_message("matbench"):
            from matbench.bench import MatbenchBenchmark  # type: ignore[reportMissingImports] # noqa

        if self.config.task is None:
            mb = MatbenchBenchmark(autoload=False)
            all_tasks = list(mb.metadata.keys())
            raise ValueError(f"Please specify a task from {all_tasks}")
        else:
            mb = MatbenchBenchmark(autoload=False, subset=[self.config.task])
            self._task = list(mb.tasks)[0]
            self._task.load()

    def _load_data(self) -> None:
        """Load and process the dataset split."""
        assert (
            self.config.fold_idx >= 0 and self.config.fold_idx < 5
        ), "Invalid fold index, should be within [0, 1, 2, 3, 4]"
        fold = self._task.folds[self.config.fold_idx]
        inputs_data, outputs_data = self._task.get_train_and_val_data(fold)

        self._atoms_list = self._convert_structures_to_atoms(inputs_data, outputs_data)
        log.info(
            f"Loaded  {len(self._atoms_list)} samples " f"(fold {self.config.fold_idx})"
        )

    def _convert_structures_to_atoms(
        self,
        structures: list[Structure],
        property_values: list[float] | None = None,
    ) -> list[ase.Atoms]:
        """Convert pymatgen structures to ASE atoms.

        Args:
            structures: List of pymatgen Structure objects.
            property_values: Optional list of property values to add to atoms.info.

        Returns:
            List of ASE ase.Atoms objects.
        """
        with optional_import_error_message("pymatgen"):
            from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore[reportMissingImports] # noqa

        adapter = AseAtomsAdaptor()
        atoms_list = []
        prop_name = (
            self.config.property_name
            if self.config.property_name is not None
            else self.config.task
        )
        for i, structure in enumerate(structures):
            atoms = adapter.get_atoms(structure)
            assert isinstance(atoms, ase.Atoms), "Expected an Atoms object"
            if property_values is not None:
                atoms.info[prop_name] = property_values[i]
            atoms_list.append(atoms)

        return atoms_list

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        """Get an item from the dataset by index."""
        return self._atoms_list[idx]

    def __len__(self) -> int:
        """Get the total number of items in the dataset."""
        return len(self._atoms_list)

    def get_test_data(self) -> list[ase.Atoms]:
        """Load the test data for the current task and fold.

        Returns:
            List of ASE ase.Atoms objects from the test set.
        """
        test_inputs = self._task.get_test_data(
            self._task.folds[self.config.fold_idx], include_target=False
        )
        return self._convert_structures_to_atoms(test_inputs)
