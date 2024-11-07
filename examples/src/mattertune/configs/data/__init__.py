from __future__ import annotations

__codegen__ = True

from mattertune.data import DatasetConfig as DatasetConfig
from mattertune.data import DatasetConfigBase as DatasetConfigBase
from mattertune.data import OMAT24DatasetConfig as OMAT24DatasetConfig
from mattertune.data import XYZDatasetConfig as XYZDatasetConfig
from mattertune.data.db import DBDatasetConfig as DBDatasetConfig
from mattertune.data.matbench import MatbenchDatasetConfig as MatbenchDatasetConfig
from mattertune.data.mp import MPDatasetConfig as MPDatasetConfig
from mattertune.data.mptraj import MPTrajDatasetConfig as MPTrajDatasetConfig

from . import base as base
from . import db as db
from . import matbench as matbench
from . import mp as mp
from . import mptraj as mptraj
from . import omat24 as omat24
from . import xyz as xyz
