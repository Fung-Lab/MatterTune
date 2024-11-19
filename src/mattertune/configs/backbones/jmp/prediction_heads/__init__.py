from __future__ import annotations

__codegen__ = True

from mattertune.backbones.jmp.prediction_heads.graph_scalar import (
    GraphScalarTargetConfig as GraphScalarTargetConfig,
)

from . import graph_scalar as graph_scalar
from .graph_scalar.GraphScalarTargetConfig_typed_dict import (
    CreateGraphScalarTargetConfig as CreateGraphScalarTargetConfig,
)
from .graph_scalar.GraphScalarTargetConfig_typed_dict import (
    GraphScalarTargetConfigTypedDict as GraphScalarTargetConfigTypedDict,
)
