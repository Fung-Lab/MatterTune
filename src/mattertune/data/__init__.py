from __future__ import annotations

from typing import Annotated, TypeAlias

from ..registry import data_registry
from .base import DatasetConfigBase
from .omat24 import OMAT24Dataset as OMAT24Dataset
from .omat24 import OMAT24DatasetConfig as OMAT24DatasetConfig
from .xyz import XYZDatasetConfig as XYZDatasetConfig

DatasetConfig: TypeAlias = Annotated[
    DatasetConfigBase,
    XYZDatasetConfig,
    data_registry.DynamicResolution(),
]
