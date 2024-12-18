from __future__ import annotations

from typing import Literal

from typing_extensions import final, override

from .base import RecipeConfigBase


@final
class NoOpRecipeConfig(RecipeConfigBase):
    """
    Example recipe that does nothing.
    """

    name: Literal["no-op"] = "no-op"
    """Discriminator for the no-op recipe."""

    @override
    def create_lightning_callback(self) -> None:
        return None
