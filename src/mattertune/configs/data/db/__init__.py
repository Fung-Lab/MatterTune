from __future__ import annotations

__codegen__ = True

from mattertune.data.db import DatasetConfigBase as DatasetConfigBase
from mattertune.data.db import DBDatasetConfig as DBDatasetConfig

__all__ = [
    "DBDatasetConfig",
    "DatasetConfigBase",
]
