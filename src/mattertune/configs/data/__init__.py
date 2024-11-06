from __future__ import annotations

__codegen__ = True

from mattertune.data import DatasetConfig as DatasetConfig
from mattertune.data import DatasetConfigBase as DatasetConfigBase
from mattertune.data import OMAT24DatasetConfig as OMAT24DatasetConfig
from mattertune.data import XYZDatasetConfig as XYZDatasetConfig
from mattertune.data.db import DBDatasetConfig as DBDatasetConfig
from mattertune.data.json import JSONDatasetConfig as JSONDatasetConfig
from mattertune.data.matbench import MatbenchDatasetConfig as MatbenchDatasetConfig

from . import base as base
from . import db as db
from . import json as json
from . import matbench as matbench
from . import omat24 as omat24
from . import xyz as xyz
