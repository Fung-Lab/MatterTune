from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated

import nshconfig as C
from lightning.pytorch.callbacks import Callback
from typing_extensions import TypeAliasType


class RecipeConfigBase(C.Config, ABC):
    """
    Base configuration for recipes.
    """

    @abstractmethod
    def create_lightning_callback(self) -> Callback | None:
        """
        Creates the PyTorch Lightning callback for this recipe, or returns
        `None` if no callback is needed.
        """
        ...

    @classmethod
    def ensure_dependencies(cls):
        """
        Ensure that all dependencies are installed.

        This method should raise an exception if any dependencies are missing,
        with a message indicating which dependencies are missing and
        how to install them.
        """
        return


recipe_registry = C.Registry(RecipeConfigBase, discriminator="name")

RecipeConfig = TypeAliasType(
    "RecipeConfig",
    Annotated[
        RecipeConfigBase,
        recipe_registry.DynamicResolution(),
    ],
)
