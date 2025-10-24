from __future__ import annotations

from typing import Literal
from abc import ABC, abstractmethod

import nshconfig as C

from lightning.pytorch.callbacks import Callback


class BaseCallbackConfig(C.Config, ABC):
    """Base configuration for MatterTune callbacks."""

    @abstractmethod
    def create_callback(self) -> Callback:
        """Creates a callback instance from this config."""
        pass