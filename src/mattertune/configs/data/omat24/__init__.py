from __future__ import annotations

__codegen__ = True

from mattertune.data.omat24 import DatasetConfigBase as DatasetConfigBase
from mattertune.data.omat24 import OMAT24DatasetConfig as OMAT24DatasetConfig
from mattertune.data.omat24 import data_registry as data_registry

__all__ = [
    "DatasetConfigBase",
    "OMAT24DatasetConfig",
    "data_registry",
]
