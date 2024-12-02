from __future__ import annotations

from typing import Literal

from typing_extensions import final

from .base import RecipeConfigBase


@final
class NoOpRecipeConfig(RecipeConfigBase):
    """
    Example recipe that does nothing.
    """

    name: Literal["no-op"] = "no-op"
    """Discriminator for the no-op recipe."""
