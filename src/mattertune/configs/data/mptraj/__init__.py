from __future__ import annotations

__codegen__ = True

from mattertune.data.mptraj import DatasetConfigBase as DatasetConfigBase
from mattertune.data.mptraj import MPTrajDatasetConfig as MPTrajDatasetConfig

from .DatasetConfigBase_typed_dict import (
    CreateDatasetConfigBase as CreateDatasetConfigBase,
)
from .DatasetConfigBase_typed_dict import (
    DatasetConfigBaseTypedDict as DatasetConfigBaseTypedDict,
)
from .MPTrajDatasetConfig_typed_dict import (
    CreateMPTrajDatasetConfig as CreateMPTrajDatasetConfig,
)
from .MPTrajDatasetConfig_typed_dict import (
    MPTrajDatasetConfigTypedDict as MPTrajDatasetConfigTypedDict,
)
