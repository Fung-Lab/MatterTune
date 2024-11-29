from __future__ import annotations

__codegen__ = True

from mattertune.backbones.eqV2.model import EqV2BackboneConfig as EqV2BackboneConfig
from mattertune.backbones.eqV2.model import (
    FAIRChemAtomsToGraphSystemConfig as FAIRChemAtomsToGraphSystemConfig,
)
from mattertune.backbones.eqV2.model import (
    FinetuneModuleBaseConfig as FinetuneModuleBaseConfig,
)

__all__ = [
    "EqV2BackboneConfig",
    "FAIRChemAtomsToGraphSystemConfig",
    "FinetuneModuleBaseConfig",
]
