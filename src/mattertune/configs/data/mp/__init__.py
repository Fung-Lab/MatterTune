from __future__ import annotations

__codegen__ = True

from mattertune.data.mp import DatasetConfigBase as DatasetConfigBase
from mattertune.data.mp import MPDatasetConfig as MPDatasetConfig

from .DatasetConfigBase_typed_dict import (
    CreateDatasetConfigBase as CreateDatasetConfigBase,
)
from .DatasetConfigBase_typed_dict import (
    DatasetConfigBaseTypedDict as DatasetConfigBaseTypedDict,
)
from .MPDatasetConfig_typed_dict import CreateMPDatasetConfig as CreateMPDatasetConfig
from .MPDatasetConfig_typed_dict import (
    MPDatasetConfigTypedDict as MPDatasetConfigTypedDict,
)
