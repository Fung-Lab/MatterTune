from __future__ import annotations

__codegen__ = True

from mattertune.data.matbench import DatasetConfigBase as DatasetConfigBase
from mattertune.data.matbench import MatbenchDatasetConfig as MatbenchDatasetConfig

from .DatasetConfigBase_typed_dict import (
    CreateDatasetConfigBase as CreateDatasetConfigBase,
)
from .DatasetConfigBase_typed_dict import (
    DatasetConfigBaseTypedDict as DatasetConfigBaseTypedDict,
)
from .MatbenchDatasetConfig_typed_dict import (
    CreateMatbenchDatasetConfig as CreateMatbenchDatasetConfig,
)
from .MatbenchDatasetConfig_typed_dict import (
    MatbenchDatasetConfigTypedDict as MatbenchDatasetConfigTypedDict,
)
