from __future__ import annotations

from typing import Annotated

from typing_extensions import TypeAliasType

from ..finetune.base import FinetuneModuleBaseConfig
from ..registry import backbone_registry
from .eqV2 import EqV2BackboneConfig as EqV2BackboneConfig
from .eqV2 import EqV2BackboneModule as EqV2BackboneModule
from .jmp import JMPBackboneConfig as JMPBackboneConfig
from .jmp import JMPBackboneModule as JMPBackboneModule
from .m3gnet import M3GNetBackboneConfig as M3GNetBackboneConfig
from .m3gnet import M3GNetBackboneModule as M3GNetBackboneModule
from .mattersim import MatterSimBackboneConfig as MatterSimBackboneConfig
from .mattersim import MatterSimM3GNetBackboneModule as MatterSimM3GNetBackboneModule
from .orb import ORBBackboneConfig as ORBBackboneConfig
from .orb import ORBBackboneModule as ORBBackboneModule

ModelConfig = TypeAliasType(
    "ModelConfig",
    Annotated[
        FinetuneModuleBaseConfig,
        backbone_registry.DynamicResolution(),
    ],
)
