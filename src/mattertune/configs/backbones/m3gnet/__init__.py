from __future__ import annotations

__codegen__ = True

from mattertune.backbones.m3gnet import M3GNetBackboneConfig as M3GNetBackboneConfig
from mattertune.backbones.m3gnet import (
    M3GNetGraphComputerConfig as M3GNetGraphComputerConfig,
)
from mattertune.backbones.m3gnet.model import (
    FinetuneModuleBaseConfig as FinetuneModuleBaseConfig,
)

from . import model as model
from .M3GNetBackboneConfig_typed_dict import (
    CreateM3GNetBackboneConfig as CreateM3GNetBackboneConfig,
)
from .M3GNetBackboneConfig_typed_dict import (
    M3GNetBackboneConfigTypedDict as M3GNetBackboneConfigTypedDict,
)
from .M3GNetGraphComputerConfig_typed_dict import (
    CreateM3GNetGraphComputerConfig as CreateM3GNetGraphComputerConfig,
)
from .M3GNetGraphComputerConfig_typed_dict import (
    M3GNetGraphComputerConfigTypedDict as M3GNetGraphComputerConfigTypedDict,
)
from .model.FinetuneModuleBaseConfig_typed_dict import (
    CreateFinetuneModuleBaseConfig as CreateFinetuneModuleBaseConfig,
)
from .model.FinetuneModuleBaseConfig_typed_dict import (
    FinetuneModuleBaseConfigTypedDict as FinetuneModuleBaseConfigTypedDict,
)
