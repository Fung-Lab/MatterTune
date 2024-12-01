from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated

import nshconfig as C
from ase import Atoms
from torch.utils.data import Dataset
from typing_extensions import TypeAliasType

from ..registry import data_registry


class DatasetConfigBase(C.Config, ABC):
    @abstractmethod
    def create_dataset(self) -> Dataset[Atoms]: ...

    def prepare_data(self):
        """
        Prepare the dataset for training.

        Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
        settings) will result in corrupted data. Lightning ensures this method is called only within a single process,
        so you can safely add your downloading logic within this method.
        """
        pass

    @classmethod
    def ensure_dependencies(cls):
        """
        Ensure that all dependencies are installed.

        This method should raise an exception if any dependencies are missing,
        with a message indicating which dependencies are missing and
        how to install them.
        """
        pass


DatasetConfig = TypeAliasType(
    "DatasetConfig",
    Annotated[DatasetConfigBase, data_registry.DynamicResolution()],
)
