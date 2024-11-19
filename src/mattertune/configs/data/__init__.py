__codegen__ = True

from mattertune.data.datamodule import AutoSplitDataModuleConfig as AutoSplitDataModuleConfig
from mattertune.data.db import DBDatasetConfig as DBDatasetConfig
from mattertune.data.datamodule import DataModuleBaseConfig as DataModuleBaseConfig
from mattertune.data import DatasetConfigBase as DatasetConfigBase
from mattertune.data.mp import MPDatasetConfig as MPDatasetConfig
from mattertune.data.mptraj import MPTrajDatasetConfig as MPTrajDatasetConfig
from mattertune.data.datamodule import ManualSplitDataModuleConfig as ManualSplitDataModuleConfig
from mattertune.data.matbench import MatbenchDatasetConfig as MatbenchDatasetConfig
from mattertune.data import OMAT24DatasetConfig as OMAT24DatasetConfig
from mattertune.data import XYZDatasetConfig as XYZDatasetConfig

from mattertune.data.datamodule import AutoSplitDataModuleConfig as AutoSplitDataModuleConfig
from mattertune.data.db import DBDatasetConfig as DBDatasetConfig
from mattertune.data.datamodule import DataModuleBaseConfig as DataModuleBaseConfig
from mattertune.data import DatasetConfigBase as DatasetConfigBase
from mattertune.data.mp import MPDatasetConfig as MPDatasetConfig
from mattertune.data.mptraj import MPTrajDatasetConfig as MPTrajDatasetConfig
from mattertune.data.datamodule import ManualSplitDataModuleConfig as ManualSplitDataModuleConfig
from mattertune.data.matbench import MatbenchDatasetConfig as MatbenchDatasetConfig
from mattertune.data import OMAT24DatasetConfig as OMAT24DatasetConfig
from mattertune.data import XYZDatasetConfig as XYZDatasetConfig

from . import base as base
from . import datamodule as datamodule
from . import db as db
from . import matbench as matbench
from . import mp as mp
from . import mptraj as mptraj
from . import omat24 as omat24
from . import xyz as xyz
