from __future__ import annotations

from abc import ABC, abstractmethod

import nshconfig as C
from ase import Atoms
from torch.utils.data import Dataset


class DatasetConfigBase(C.Config, ABC):
    @abstractmethod
    def create_dataset(self) -> Dataset[Atoms]: ...

    @classmethod
    def ensure_dependencies(cls):
        """
        Ensure that all dependencies are installed.

        This method should raise an exception if any dependencies are missing,
            with a message indicating which dependencies are missing and
            how to install them.
        """
        pass
