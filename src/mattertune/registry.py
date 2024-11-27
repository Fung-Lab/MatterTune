from __future__ import annotations

import nshconfig as C

from .finetune.base import FinetuneModuleBaseConfig

backbone_registry = C.Registry(FinetuneModuleBaseConfig, discriminator="name")
"""Registry for backbone modules."""

data_registry = C.Registry(C.Config, discriminator="type")
"""Registry for data modules."""
__all__ = [
    "backbone_registry",
    "data_registry",
]
