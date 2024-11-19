from __future__ import annotations

__codegen__ = True

from mattertune.finetune.loss import HuberLossConfig as HuberLossConfig
from mattertune.finetune.loss import L2MAELossConfig as L2MAELossConfig
from mattertune.finetune.loss import MAELossConfig as MAELossConfig
from mattertune.finetune.loss import MSELossConfig as MSELossConfig

from .HuberLossConfig_typed_dict import CreateHuberLossConfig as CreateHuberLossConfig
from .HuberLossConfig_typed_dict import (
    HuberLossConfigTypedDict as HuberLossConfigTypedDict,
)
from .L2MAELossConfig_typed_dict import CreateL2MAELossConfig as CreateL2MAELossConfig
from .L2MAELossConfig_typed_dict import (
    L2MAELossConfigTypedDict as L2MAELossConfigTypedDict,
)
from .MAELossConfig_typed_dict import CreateMAELossConfig as CreateMAELossConfig
from .MAELossConfig_typed_dict import MAELossConfigTypedDict as MAELossConfigTypedDict
from .MSELossConfig_typed_dict import CreateMSELossConfig as CreateMSELossConfig
from .MSELossConfig_typed_dict import MSELossConfigTypedDict as MSELossConfigTypedDict
