from __future__ import annotations

__codegen__ = True

from mattertune.loggers import CSVLoggerConfig as CSVLoggerConfig
from mattertune.loggers import TensorBoardLoggerConfig as TensorBoardLoggerConfig
from mattertune.loggers import WandbLoggerConfig as WandbLoggerConfig

from .CSVLoggerConfig_typed_dict import CreateCSVLoggerConfig as CreateCSVLoggerConfig
from .CSVLoggerConfig_typed_dict import (
    CSVLoggerConfigTypedDict as CSVLoggerConfigTypedDict,
)
from .TensorBoardLoggerConfig_typed_dict import (
    CreateTensorBoardLoggerConfig as CreateTensorBoardLoggerConfig,
)
from .TensorBoardLoggerConfig_typed_dict import (
    TensorBoardLoggerConfigTypedDict as TensorBoardLoggerConfigTypedDict,
)
from .WandbLoggerConfig_typed_dict import (
    CreateWandbLoggerConfig as CreateWandbLoggerConfig,
)
from .WandbLoggerConfig_typed_dict import (
    WandbLoggerConfigTypedDict as WandbLoggerConfigTypedDict,
)
