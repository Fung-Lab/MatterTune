from __future__ import annotations

__codegen__ = True

from mattertune.backbones.jmp import JMPBackboneConfig as JMPBackboneConfig
from mattertune.backbones.jmp.model import CutoffsConfig as CutoffsConfig
from mattertune.backbones.jmp.model import (
    FinetuneModuleBaseConfig as FinetuneModuleBaseConfig,
)
from mattertune.backbones.jmp.model import (
    JMPGraphComputerConfig as JMPGraphComputerConfig,
)
from mattertune.backbones.jmp.model import MaxNeighborsConfig as MaxNeighborsConfig
from mattertune.backbones.jmp.prediction_heads.graph_scalar import (
    GraphScalarTargetConfig as GraphScalarTargetConfig,
)

from . import model as model
from . import prediction_heads as prediction_heads
from .JMPBackboneConfig_typed_dict import (
    CreateJMPBackboneConfig as CreateJMPBackboneConfig,
)
from .JMPBackboneConfig_typed_dict import (
    JMPBackboneConfigTypedDict as JMPBackboneConfigTypedDict,
)
from .model.CutoffsConfig_typed_dict import CreateCutoffsConfig as CreateCutoffsConfig
from .model.CutoffsConfig_typed_dict import (
    CutoffsConfigTypedDict as CutoffsConfigTypedDict,
)
from .model.FinetuneModuleBaseConfig_typed_dict import (
    CreateFinetuneModuleBaseConfig as CreateFinetuneModuleBaseConfig,
)
from .model.FinetuneModuleBaseConfig_typed_dict import (
    FinetuneModuleBaseConfigTypedDict as FinetuneModuleBaseConfigTypedDict,
)
from .model.JMPGraphComputerConfig_typed_dict import (
    CreateJMPGraphComputerConfig as CreateJMPGraphComputerConfig,
)
from .model.JMPGraphComputerConfig_typed_dict import (
    JMPGraphComputerConfigTypedDict as JMPGraphComputerConfigTypedDict,
)
from .model.MaxNeighborsConfig_typed_dict import (
    CreateMaxNeighborsConfig as CreateMaxNeighborsConfig,
)
from .model.MaxNeighborsConfig_typed_dict import (
    MaxNeighborsConfigTypedDict as MaxNeighborsConfigTypedDict,
)
from .prediction_heads.graph_scalar.GraphScalarTargetConfig_typed_dict import (
    CreateGraphScalarTargetConfig as CreateGraphScalarTargetConfig,
)
from .prediction_heads.graph_scalar.GraphScalarTargetConfig_typed_dict import (
    GraphScalarTargetConfigTypedDict as GraphScalarTargetConfigTypedDict,
)
