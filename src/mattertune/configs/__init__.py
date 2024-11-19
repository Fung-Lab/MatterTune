from __future__ import annotations

__codegen__ = True

from mattertune.backbones import EqV2BackboneConfig as EqV2BackboneConfig
from mattertune.backbones import JMPBackboneConfig as JMPBackboneConfig
from mattertune.backbones import M3GNetBackboneConfig as M3GNetBackboneConfig
from mattertune.backbones import ORBBackboneConfig as ORBBackboneConfig
from mattertune.backbones.eqV2.model import (
    FAIRChemAtomsToGraphSystemConfig as FAIRChemAtomsToGraphSystemConfig,
)
from mattertune.backbones.jmp.model import CutoffsConfig as CutoffsConfig
from mattertune.backbones.jmp.model import (
    JMPGraphComputerConfig as JMPGraphComputerConfig,
)
from mattertune.backbones.jmp.model import MaxNeighborsConfig as MaxNeighborsConfig
from mattertune.backbones.jmp.prediction_heads.graph_scalar import (
    GraphScalarTargetConfig as GraphScalarTargetConfig,
)
from mattertune.backbones.m3gnet import (
    M3GNetGraphComputerConfig as M3GNetGraphComputerConfig,
)
from mattertune.backbones.orb.model import ORBSystemConfig as ORBSystemConfig
from mattertune.data import DatasetConfigBase as DatasetConfigBase
from mattertune.data import OMAT24DatasetConfig as OMAT24DatasetConfig
from mattertune.data import XYZDatasetConfig as XYZDatasetConfig
from mattertune.data.datamodule import (
    AutoSplitDataModuleConfig as AutoSplitDataModuleConfig,
)
from mattertune.data.datamodule import DataModuleBaseConfig as DataModuleBaseConfig
from mattertune.data.datamodule import (
    ManualSplitDataModuleConfig as ManualSplitDataModuleConfig,
)
from mattertune.data.db import DBDatasetConfig as DBDatasetConfig
from mattertune.data.matbench import MatbenchDatasetConfig as MatbenchDatasetConfig
from mattertune.data.mp import MPDatasetConfig as MPDatasetConfig
from mattertune.data.mptraj import MPTrajDatasetConfig as MPTrajDatasetConfig
from mattertune.finetune.loss import HuberLossConfig as HuberLossConfig
from mattertune.finetune.loss import L2MAELossConfig as L2MAELossConfig
from mattertune.finetune.loss import MAELossConfig as MAELossConfig
from mattertune.finetune.loss import MSELossConfig as MSELossConfig
from mattertune.finetune.lr_scheduler import (
    CosineAnnealingLRConfig as CosineAnnealingLRConfig,
)
from mattertune.finetune.lr_scheduler import ExponentialConfig as ExponentialConfig
from mattertune.finetune.lr_scheduler import MultiStepLRConfig as MultiStepLRConfig
from mattertune.finetune.lr_scheduler import (
    ReduceOnPlateauConfig as ReduceOnPlateauConfig,
)
from mattertune.finetune.lr_scheduler import StepLRConfig as StepLRConfig
from mattertune.finetune.optimizer import AdamConfig as AdamConfig
from mattertune.finetune.optimizer import AdamWConfig as AdamWConfig
from mattertune.finetune.optimizer import SGDConfig as SGDConfig
from mattertune.finetune.properties import EnergyPropertyConfig as EnergyPropertyConfig
from mattertune.finetune.properties import ForcesPropertyConfig as ForcesPropertyConfig
from mattertune.finetune.properties import GraphPropertyConfig as GraphPropertyConfig
from mattertune.finetune.properties import PropertyConfigBase as PropertyConfigBase
from mattertune.finetune.properties import (
    StressesPropertyConfig as StressesPropertyConfig,
)
from mattertune.loggers import CSVLoggerConfig as CSVLoggerConfig
from mattertune.loggers import TensorBoardLoggerConfig as TensorBoardLoggerConfig
from mattertune.loggers import WandbLoggerConfig as WandbLoggerConfig
from mattertune.main import EarlyStoppingConfig as EarlyStoppingConfig
from mattertune.main import MatterTunerConfig as MatterTunerConfig
from mattertune.main import ModelCheckpointConfig as ModelCheckpointConfig
from mattertune.main import TrainerConfig as TrainerConfig
from mattertune.normalization import MeanStdNormalizerConfig as MeanStdNormalizerConfig
from mattertune.normalization import NormalizerConfigBase as NormalizerConfigBase
from mattertune.normalization import (
    PerAtomReferencingNormalizerConfig as PerAtomReferencingNormalizerConfig,
)
from mattertune.normalization import RMSNormalizerConfig as RMSNormalizerConfig
from mattertune.registry import FinetuneModuleBaseConfig as FinetuneModuleBaseConfig

from . import backbones as backbones
from . import callbacks as callbacks
from . import data as data
from . import finetune as finetune
from . import loggers as loggers
from . import main as main
from . import normalization as normalization
from . import registry as registry
from .backbones.eqV2.model.FAIRChemAtomsToGraphSystemConfig_typed_dict import (
    CreateFAIRChemAtomsToGraphSystemConfig as CreateFAIRChemAtomsToGraphSystemConfig,
)
from .backbones.eqV2.model.FAIRChemAtomsToGraphSystemConfig_typed_dict import (
    FAIRChemAtomsToGraphSystemConfigTypedDict as FAIRChemAtomsToGraphSystemConfigTypedDict,
)
from .backbones.EqV2BackboneConfig_typed_dict import (
    CreateEqV2BackboneConfig as CreateEqV2BackboneConfig,
)
from .backbones.EqV2BackboneConfig_typed_dict import (
    EqV2BackboneConfigTypedDict as EqV2BackboneConfigTypedDict,
)
from .backbones.jmp.model.CutoffsConfig_typed_dict import (
    CreateCutoffsConfig as CreateCutoffsConfig,
)
from .backbones.jmp.model.CutoffsConfig_typed_dict import (
    CutoffsConfigTypedDict as CutoffsConfigTypedDict,
)
from .backbones.jmp.model.JMPGraphComputerConfig_typed_dict import (
    CreateJMPGraphComputerConfig as CreateJMPGraphComputerConfig,
)
from .backbones.jmp.model.JMPGraphComputerConfig_typed_dict import (
    JMPGraphComputerConfigTypedDict as JMPGraphComputerConfigTypedDict,
)
from .backbones.jmp.model.MaxNeighborsConfig_typed_dict import (
    CreateMaxNeighborsConfig as CreateMaxNeighborsConfig,
)
from .backbones.jmp.model.MaxNeighborsConfig_typed_dict import (
    MaxNeighborsConfigTypedDict as MaxNeighborsConfigTypedDict,
)
from .backbones.jmp.prediction_heads.graph_scalar.GraphScalarTargetConfig_typed_dict import (
    CreateGraphScalarTargetConfig as CreateGraphScalarTargetConfig,
)
from .backbones.jmp.prediction_heads.graph_scalar.GraphScalarTargetConfig_typed_dict import (
    GraphScalarTargetConfigTypedDict as GraphScalarTargetConfigTypedDict,
)
from .backbones.JMPBackboneConfig_typed_dict import (
    CreateJMPBackboneConfig as CreateJMPBackboneConfig,
)
from .backbones.JMPBackboneConfig_typed_dict import (
    JMPBackboneConfigTypedDict as JMPBackboneConfigTypedDict,
)
from .backbones.m3gnet.M3GNetGraphComputerConfig_typed_dict import (
    CreateM3GNetGraphComputerConfig as CreateM3GNetGraphComputerConfig,
)
from .backbones.m3gnet.M3GNetGraphComputerConfig_typed_dict import (
    M3GNetGraphComputerConfigTypedDict as M3GNetGraphComputerConfigTypedDict,
)
from .backbones.M3GNetBackboneConfig_typed_dict import (
    CreateM3GNetBackboneConfig as CreateM3GNetBackboneConfig,
)
from .backbones.M3GNetBackboneConfig_typed_dict import (
    M3GNetBackboneConfigTypedDict as M3GNetBackboneConfigTypedDict,
)
from .backbones.orb.model.ORBSystemConfig_typed_dict import (
    CreateORBSystemConfig as CreateORBSystemConfig,
)
from .backbones.orb.model.ORBSystemConfig_typed_dict import (
    ORBSystemConfigTypedDict as ORBSystemConfigTypedDict,
)
from .backbones.ORBBackboneConfig_typed_dict import (
    CreateORBBackboneConfig as CreateORBBackboneConfig,
)
from .backbones.ORBBackboneConfig_typed_dict import (
    ORBBackboneConfigTypedDict as ORBBackboneConfigTypedDict,
)
from .data.datamodule.AutoSplitDataModuleConfig_typed_dict import (
    AutoSplitDataModuleConfigTypedDict as AutoSplitDataModuleConfigTypedDict,
)
from .data.datamodule.AutoSplitDataModuleConfig_typed_dict import (
    CreateAutoSplitDataModuleConfig as CreateAutoSplitDataModuleConfig,
)
from .data.datamodule.DataModuleBaseConfig_typed_dict import (
    CreateDataModuleBaseConfig as CreateDataModuleBaseConfig,
)
from .data.datamodule.DataModuleBaseConfig_typed_dict import (
    DataModuleBaseConfigTypedDict as DataModuleBaseConfigTypedDict,
)
from .data.datamodule.ManualSplitDataModuleConfig_typed_dict import (
    CreateManualSplitDataModuleConfig as CreateManualSplitDataModuleConfig,
)
from .data.datamodule.ManualSplitDataModuleConfig_typed_dict import (
    ManualSplitDataModuleConfigTypedDict as ManualSplitDataModuleConfigTypedDict,
)
from .data.DatasetConfigBase_typed_dict import (
    CreateDatasetConfigBase as CreateDatasetConfigBase,
)
from .data.DatasetConfigBase_typed_dict import (
    DatasetConfigBaseTypedDict as DatasetConfigBaseTypedDict,
)
from .data.db.DBDatasetConfig_typed_dict import (
    CreateDBDatasetConfig as CreateDBDatasetConfig,
)
from .data.db.DBDatasetConfig_typed_dict import (
    DBDatasetConfigTypedDict as DBDatasetConfigTypedDict,
)
from .data.matbench.MatbenchDatasetConfig_typed_dict import (
    CreateMatbenchDatasetConfig as CreateMatbenchDatasetConfig,
)
from .data.matbench.MatbenchDatasetConfig_typed_dict import (
    MatbenchDatasetConfigTypedDict as MatbenchDatasetConfigTypedDict,
)
from .data.mp.MPDatasetConfig_typed_dict import (
    CreateMPDatasetConfig as CreateMPDatasetConfig,
)
from .data.mp.MPDatasetConfig_typed_dict import (
    MPDatasetConfigTypedDict as MPDatasetConfigTypedDict,
)
from .data.mptraj.MPTrajDatasetConfig_typed_dict import (
    CreateMPTrajDatasetConfig as CreateMPTrajDatasetConfig,
)
from .data.mptraj.MPTrajDatasetConfig_typed_dict import (
    MPTrajDatasetConfigTypedDict as MPTrajDatasetConfigTypedDict,
)
from .data.OMAT24DatasetConfig_typed_dict import (
    CreateOMAT24DatasetConfig as CreateOMAT24DatasetConfig,
)
from .data.OMAT24DatasetConfig_typed_dict import (
    OMAT24DatasetConfigTypedDict as OMAT24DatasetConfigTypedDict,
)
from .data.XYZDatasetConfig_typed_dict import (
    CreateXYZDatasetConfig as CreateXYZDatasetConfig,
)
from .data.XYZDatasetConfig_typed_dict import (
    XYZDatasetConfigTypedDict as XYZDatasetConfigTypedDict,
)
from .finetune.loss.HuberLossConfig_typed_dict import (
    CreateHuberLossConfig as CreateHuberLossConfig,
)
from .finetune.loss.HuberLossConfig_typed_dict import (
    HuberLossConfigTypedDict as HuberLossConfigTypedDict,
)
from .finetune.loss.L2MAELossConfig_typed_dict import (
    CreateL2MAELossConfig as CreateL2MAELossConfig,
)
from .finetune.loss.L2MAELossConfig_typed_dict import (
    L2MAELossConfigTypedDict as L2MAELossConfigTypedDict,
)
from .finetune.loss.MAELossConfig_typed_dict import (
    CreateMAELossConfig as CreateMAELossConfig,
)
from .finetune.loss.MAELossConfig_typed_dict import (
    MAELossConfigTypedDict as MAELossConfigTypedDict,
)
from .finetune.loss.MSELossConfig_typed_dict import (
    CreateMSELossConfig as CreateMSELossConfig,
)
from .finetune.loss.MSELossConfig_typed_dict import (
    MSELossConfigTypedDict as MSELossConfigTypedDict,
)
from .finetune.lr_scheduler.CosineAnnealingLRConfig_typed_dict import (
    CosineAnnealingLRConfigTypedDict as CosineAnnealingLRConfigTypedDict,
)
from .finetune.lr_scheduler.CosineAnnealingLRConfig_typed_dict import (
    CreateCosineAnnealingLRConfig as CreateCosineAnnealingLRConfig,
)
from .finetune.lr_scheduler.ExponentialConfig_typed_dict import (
    CreateExponentialConfig as CreateExponentialConfig,
)
from .finetune.lr_scheduler.ExponentialConfig_typed_dict import (
    ExponentialConfigTypedDict as ExponentialConfigTypedDict,
)
from .finetune.lr_scheduler.MultiStepLRConfig_typed_dict import (
    CreateMultiStepLRConfig as CreateMultiStepLRConfig,
)
from .finetune.lr_scheduler.MultiStepLRConfig_typed_dict import (
    MultiStepLRConfigTypedDict as MultiStepLRConfigTypedDict,
)
from .finetune.lr_scheduler.ReduceOnPlateauConfig_typed_dict import (
    CreateReduceOnPlateauConfig as CreateReduceOnPlateauConfig,
)
from .finetune.lr_scheduler.ReduceOnPlateauConfig_typed_dict import (
    ReduceOnPlateauConfigTypedDict as ReduceOnPlateauConfigTypedDict,
)
from .finetune.lr_scheduler.StepLRConfig_typed_dict import (
    CreateStepLRConfig as CreateStepLRConfig,
)
from .finetune.lr_scheduler.StepLRConfig_typed_dict import (
    StepLRConfigTypedDict as StepLRConfigTypedDict,
)
from .finetune.optimizer.AdamConfig_typed_dict import (
    AdamConfigTypedDict as AdamConfigTypedDict,
)
from .finetune.optimizer.AdamConfig_typed_dict import (
    CreateAdamConfig as CreateAdamConfig,
)
from .finetune.optimizer.AdamWConfig_typed_dict import (
    AdamWConfigTypedDict as AdamWConfigTypedDict,
)
from .finetune.optimizer.AdamWConfig_typed_dict import (
    CreateAdamWConfig as CreateAdamWConfig,
)
from .finetune.optimizer.SGDConfig_typed_dict import CreateSGDConfig as CreateSGDConfig
from .finetune.optimizer.SGDConfig_typed_dict import (
    SGDConfigTypedDict as SGDConfigTypedDict,
)
from .finetune.properties.EnergyPropertyConfig_typed_dict import (
    CreateEnergyPropertyConfig as CreateEnergyPropertyConfig,
)
from .finetune.properties.EnergyPropertyConfig_typed_dict import (
    EnergyPropertyConfigTypedDict as EnergyPropertyConfigTypedDict,
)
from .finetune.properties.ForcesPropertyConfig_typed_dict import (
    CreateForcesPropertyConfig as CreateForcesPropertyConfig,
)
from .finetune.properties.ForcesPropertyConfig_typed_dict import (
    ForcesPropertyConfigTypedDict as ForcesPropertyConfigTypedDict,
)
from .finetune.properties.GraphPropertyConfig_typed_dict import (
    CreateGraphPropertyConfig as CreateGraphPropertyConfig,
)
from .finetune.properties.GraphPropertyConfig_typed_dict import (
    GraphPropertyConfigTypedDict as GraphPropertyConfigTypedDict,
)
from .finetune.properties.PropertyConfigBase_typed_dict import (
    CreatePropertyConfigBase as CreatePropertyConfigBase,
)
from .finetune.properties.PropertyConfigBase_typed_dict import (
    PropertyConfigBaseTypedDict as PropertyConfigBaseTypedDict,
)
from .finetune.properties.StressesPropertyConfig_typed_dict import (
    CreateStressesPropertyConfig as CreateStressesPropertyConfig,
)
from .finetune.properties.StressesPropertyConfig_typed_dict import (
    StressesPropertyConfigTypedDict as StressesPropertyConfigTypedDict,
)
from .loggers.CSVLoggerConfig_typed_dict import (
    CreateCSVLoggerConfig as CreateCSVLoggerConfig,
)
from .loggers.CSVLoggerConfig_typed_dict import (
    CSVLoggerConfigTypedDict as CSVLoggerConfigTypedDict,
)
from .loggers.TensorBoardLoggerConfig_typed_dict import (
    CreateTensorBoardLoggerConfig as CreateTensorBoardLoggerConfig,
)
from .loggers.TensorBoardLoggerConfig_typed_dict import (
    TensorBoardLoggerConfigTypedDict as TensorBoardLoggerConfigTypedDict,
)
from .loggers.WandbLoggerConfig_typed_dict import (
    CreateWandbLoggerConfig as CreateWandbLoggerConfig,
)
from .loggers.WandbLoggerConfig_typed_dict import (
    WandbLoggerConfigTypedDict as WandbLoggerConfigTypedDict,
)
from .main.EarlyStoppingConfig_typed_dict import (
    CreateEarlyStoppingConfig as CreateEarlyStoppingConfig,
)
from .main.EarlyStoppingConfig_typed_dict import (
    EarlyStoppingConfigTypedDict as EarlyStoppingConfigTypedDict,
)
from .main.MatterTunerConfig_typed_dict import (
    CreateMatterTunerConfig as CreateMatterTunerConfig,
)
from .main.MatterTunerConfig_typed_dict import (
    MatterTunerConfigTypedDict as MatterTunerConfigTypedDict,
)
from .main.ModelCheckpointConfig_typed_dict import (
    CreateModelCheckpointConfig as CreateModelCheckpointConfig,
)
from .main.ModelCheckpointConfig_typed_dict import (
    ModelCheckpointConfigTypedDict as ModelCheckpointConfigTypedDict,
)
from .main.TrainerConfig_typed_dict import CreateTrainerConfig as CreateTrainerConfig
from .main.TrainerConfig_typed_dict import (
    TrainerConfigTypedDict as TrainerConfigTypedDict,
)
from .normalization.MeanStdNormalizerConfig_typed_dict import (
    CreateMeanStdNormalizerConfig as CreateMeanStdNormalizerConfig,
)
from .normalization.MeanStdNormalizerConfig_typed_dict import (
    MeanStdNormalizerConfigTypedDict as MeanStdNormalizerConfigTypedDict,
)
from .normalization.NormalizerConfigBase_typed_dict import (
    CreateNormalizerConfigBase as CreateNormalizerConfigBase,
)
from .normalization.NormalizerConfigBase_typed_dict import (
    NormalizerConfigBaseTypedDict as NormalizerConfigBaseTypedDict,
)
from .normalization.PerAtomReferencingNormalizerConfig_typed_dict import (
    CreatePerAtomReferencingNormalizerConfig as CreatePerAtomReferencingNormalizerConfig,
)
from .normalization.PerAtomReferencingNormalizerConfig_typed_dict import (
    PerAtomReferencingNormalizerConfigTypedDict as PerAtomReferencingNormalizerConfigTypedDict,
)
from .normalization.RMSNormalizerConfig_typed_dict import (
    CreateRMSNormalizerConfig as CreateRMSNormalizerConfig,
)
from .normalization.RMSNormalizerConfig_typed_dict import (
    RMSNormalizerConfigTypedDict as RMSNormalizerConfigTypedDict,
)
from .registry.FinetuneModuleBaseConfig_typed_dict import (
    CreateFinetuneModuleBaseConfig as CreateFinetuneModuleBaseConfig,
)
from .registry.FinetuneModuleBaseConfig_typed_dict import (
    FinetuneModuleBaseConfigTypedDict as FinetuneModuleBaseConfigTypedDict,
)
