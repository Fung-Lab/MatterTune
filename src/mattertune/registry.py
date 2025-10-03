from __future__ import annotations

import nshconfig as C

from .finetune.base import FinetuneModuleBaseConfig
from .distillation.base import StudentModuleBaseConfig

backbone_registry = C.Registry(FinetuneModuleBaseConfig, discriminator="name")
"""Registry for backbone modules."""

student_registry = C.Registry(StudentModuleBaseConfig, discriminator="name")

data_registry = C.Registry(C.Config, discriminator="type")
"""Registry for data modules."""
__all__ = [
    "backbone_registry",
    "student_registry",
    "data_registry",
]
