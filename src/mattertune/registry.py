from __future__ import annotations

import nshconfig as C

from .finetune.base import FinetuneModuleBaseConfig

backbone_registry = C.Registry(FinetuneModuleBaseConfig, discriminator="name")
data_registry = C.Registry(C.Config, discriminator="type")
