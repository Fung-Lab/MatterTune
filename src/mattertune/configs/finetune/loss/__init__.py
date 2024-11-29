from __future__ import annotations

__codegen__ = True

from mattertune.finetune.loss import HuberLossConfig as HuberLossConfig
from mattertune.finetune.loss import L2MAELossConfig as L2MAELossConfig
from mattertune.finetune.loss import MAELossConfig as MAELossConfig
from mattertune.finetune.loss import MSELossConfig as MSELossConfig

__all__ = [
    "HuberLossConfig",
    "L2MAELossConfig",
    "MAELossConfig",
    "MSELossConfig",
]
