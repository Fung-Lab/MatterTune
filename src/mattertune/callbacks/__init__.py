from __future__ import annotations

from typing import Annotated, Literal
from typing_extensions import TypeAliasType

import nshconfig as C

from .base import BaseCallbackConfig
from .early_stopping import EarlyStoppingConfig
from .model_checkpoint import ModelCheckpointConfig
from .per_layer_dynamics import PerLayerTrainDynamicsConfig

CallBackConfigs = TypeAliasType(
    "CallBackConfigs",
    Annotated[
        EarlyStoppingConfig 
        | ModelCheckpointConfig
        | PerLayerTrainDynamicsConfig,
        C.Field(
            description="Configuration for a MatterTune callback.",
            discriminator="name",
        ),
    ],
)

from .ema import EMAConfig
from .multi_gpu_writer import CustomWriter