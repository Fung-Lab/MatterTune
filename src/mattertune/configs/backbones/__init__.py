from __future__ import annotations

__codegen__ = True

from mattertune.backbones import EqV2BackboneConfig as EqV2BackboneConfig
from mattertune.backbones import FinetuneModuleBaseConfig as FinetuneModuleBaseConfig
from mattertune.backbones import JMPBackboneConfig as JMPBackboneConfig
from mattertune.backbones import M3GNetBackboneConfig as M3GNetBackboneConfig
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
from .eqV2.model.FAIRChemAtomsToGraphSystemConfig_typed_dict import (
    CreateFAIRChemAtomsToGraphSystemConfig as CreateFAIRChemAtomsToGraphSystemConfig,
)
from .eqV2.model.FAIRChemAtomsToGraphSystemConfig_typed_dict import (
    FAIRChemAtomsToGraphSystemConfigTypedDict as FAIRChemAtomsToGraphSystemConfigTypedDict,
)
from .EqV2BackboneConfig_typed_dict import (
    CreateEqV2BackboneConfig as CreateEqV2BackboneConfig,
)
from .EqV2BackboneConfig_typed_dict import (
    EqV2BackboneConfigTypedDict as EqV2BackboneConfigTypedDict,
)
from .FinetuneModuleBaseConfig_typed_dict import (
    CreateFinetuneModuleBaseConfig as CreateFinetuneModuleBaseConfig,
)
from .FinetuneModuleBaseConfig_typed_dict import (
    FinetuneModuleBaseConfigTypedDict as FinetuneModuleBaseConfigTypedDict,
)
from .jmp.model.CutoffsConfig_typed_dict import (
    CreateCutoffsConfig as CreateCutoffsConfig,
)
from .jmp.model.CutoffsConfig_typed_dict import (
    CutoffsConfigTypedDict as CutoffsConfigTypedDict,
)
from .jmp.model.JMPGraphComputerConfig_typed_dict import (
    CreateJMPGraphComputerConfig as CreateJMPGraphComputerConfig,
)
from .jmp.model.JMPGraphComputerConfig_typed_dict import (
    JMPGraphComputerConfigTypedDict as JMPGraphComputerConfigTypedDict,
)
from .jmp.model.MaxNeighborsConfig_typed_dict import (
    CreateMaxNeighborsConfig as CreateMaxNeighborsConfig,
)
from .jmp.model.MaxNeighborsConfig_typed_dict import (
    MaxNeighborsConfigTypedDict as MaxNeighborsConfigTypedDict,
)
from .jmp.prediction_heads.graph_scalar.GraphScalarTargetConfig_typed_dict import (
    CreateGraphScalarTargetConfig as CreateGraphScalarTargetConfig,
)
from .jmp.prediction_heads.graph_scalar.GraphScalarTargetConfig_typed_dict import (
    GraphScalarTargetConfigTypedDict as GraphScalarTargetConfigTypedDict,
)
from .JMPBackboneConfig_typed_dict import (
    CreateJMPBackboneConfig as CreateJMPBackboneConfig,
)
from .JMPBackboneConfig_typed_dict import (
    JMPBackboneConfigTypedDict as JMPBackboneConfigTypedDict,
)
from .m3gnet.M3GNetGraphComputerConfig_typed_dict import (
    CreateM3GNetGraphComputerConfig as CreateM3GNetGraphComputerConfig,
)
from .m3gnet.M3GNetGraphComputerConfig_typed_dict import (
    M3GNetGraphComputerConfigTypedDict as M3GNetGraphComputerConfigTypedDict,
)
from .M3GNetBackboneConfig_typed_dict import (
    CreateM3GNetBackboneConfig as CreateM3GNetBackboneConfig,
)
from .M3GNetBackboneConfig_typed_dict import (
    M3GNetBackboneConfigTypedDict as M3GNetBackboneConfigTypedDict,
)
from .orb.model.ORBSystemConfig_typed_dict import (
    CreateORBSystemConfig as CreateORBSystemConfig,
)
from .orb.model.ORBSystemConfig_typed_dict import (
    ORBSystemConfigTypedDict as ORBSystemConfigTypedDict,
)
from .ORBBackboneConfig_typed_dict import (
    CreateORBBackboneConfig as CreateORBBackboneConfig,
)
from .ORBBackboneConfig_typed_dict import (
    ORBBackboneConfigTypedDict as ORBBackboneConfigTypedDict,
)
