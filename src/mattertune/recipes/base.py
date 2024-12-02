from __future__ import annotations

import nshconfig as C
from lightning.pytorch.callbacks import Callback


class RecipeConfigBase(C.Config):
    """
    Base configuration for recipes.
    """

    def create_lightning_callback(self) -> Callback | None:
        """
        Creates the PyTorch Lightning callback for this recipe, or returns
        `None` if no callback is needed.
        """
        return None

    @classmethod
    def ensure_dependencies(cls):
        """
        Ensure that all dependencies are installed.

        This method should raise an exception if any dependencies are missing,
        with a message indicating which dependencies are missing and
        how to install them.
        """
        return
