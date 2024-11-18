from __future__ import annotations

__codegen__ = True

from mattertune.backbones import EqV2BackboneConfig as EqV2BackboneConfig
from mattertune.backbones import FinetuneModuleBaseConfig as FinetuneModuleBaseConfig
from mattertune.backbones import JMPBackboneConfig as JMPBackboneConfig
from mattertune.backbones import M3GNetBackboneConfig as M3GNetBackboneConfig
from mattertune.backbones import ModelConfig as ModelConfig
from mattertune.backbones import ORBBackboneConfig as ORBBackboneConfig
from mattertune.backbones.eqV2.model import (
    FAIRChemAtomsToGraphSystemConfig as FAIRChemAtomsToGraphSystemConfig,
)
from mattertune.backbones.jmp.model import CutoffsConfig as CutoffsConfig
from mattertune.backbones.jmp.model import (
    JMPGraphComputerConfig as JMPGraphComputerConfig,
)
from mattertune.backbones.jmp.model import MaxNeighborsConfig as MaxNeighborsConfig
from mattertune.backbones.jmp.prediction_heads.graph_scalar import (
    GraphScalarTargetConfig as GraphScalarTargetConfig,
)
from mattertune.backbones.m3gnet import (
    M3GNetGraphComputerConfig as M3GNetGraphComputerConfig,
)
from mattertune.backbones.orb.model import ORBSystemConfig as ORBSystemConfig

from . import eqV2 as eqV2
from . import jmp as jmp
from . import m3gnet as m3gnet
from . import orb as orb
