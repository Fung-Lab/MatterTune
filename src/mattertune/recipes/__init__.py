from __future__ import annotations

from typing import Annotated

import nshconfig as C
from typing_extensions import TypeAliasType

from .lora import LoRARecipeConfig as LoRARecipeConfig
from .noop import NoOpRecipeConfig as NoOpRecipeConfig

RecipeConfig = TypeAliasType(
    "RecipeConfig",
    Annotated[LoRARecipeConfig | NoOpRecipeConfig, C.Field(discriminator="name")],
)
