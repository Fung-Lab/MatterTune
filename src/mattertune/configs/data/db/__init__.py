from __future__ import annotations

__codegen__ = True

from mattertune.data.db import DatasetConfigBase as DatasetConfigBase
from mattertune.data.db import DBDatasetConfig as DBDatasetConfig

from .DatasetConfigBase_typed_dict import (
    CreateDatasetConfigBase as CreateDatasetConfigBase,
)
from .DatasetConfigBase_typed_dict import (
    DatasetConfigBaseTypedDict as DatasetConfigBaseTypedDict,
)
from .DBDatasetConfig_typed_dict import CreateDBDatasetConfig as CreateDBDatasetConfig
from .DBDatasetConfig_typed_dict import (
    DBDatasetConfigTypedDict as DBDatasetConfigTypedDict,
)
