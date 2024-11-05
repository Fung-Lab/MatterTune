from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

import nshconfig as C
import torch
import torch.nn as nn
from typing_extensions import assert_never


class AdamConfig(C.Config):
    name: Literal["Adam"] = "Adam"
    """name of the optimizer."""
    lr: C.PositiveFloat
    """Learning rate."""
    eps: C.NonNegativeFloat = 1e-8
    """Epsilon."""
    betas: tuple[C.PositiveFloat, C.PositiveFloat] = (0.9, 0.999)
    """Betas."""
    weight_decay: C.NonNegativeFloat = 0.0
    """Weight decay."""
    amsgrad: bool = False
    """Whether to use AMSGrad variant of Adam."""


class AdamWConfig(C.Config):
    name: Literal["AdamW"] = "AdamW"
    """name of the optimizer."""
    lr: C.PositiveFloat
    """Learning rate."""
    eps: C.NonNegativeFloat = 1e-8
    """Epsilon."""
    betas: tuple[C.PositiveFloat, C.PositiveFloat] = (0.9, 0.999)
    """Betas."""
    weight_decay: C.NonNegativeFloat = 0.01
    """Weight decay."""
    amsgrad: bool = False
    """Whether to use AMSGrad variant of Adam."""


class SGDConfig(C.Config):
    name: Literal["SGD"] = "SGD"
    """name of the optimizer."""
    lr: C.PositiveFloat
    """Learning rate."""
    momentum: C.NonNegativeFloat = 0.0
    """Momentum."""
    weight_decay: C.NonNegativeFloat = 0.0
    """Weight decay."""
    nestrov: bool = False
    """Whether to use nestrov."""


OptimizerConfig: TypeAlias = Annotated[
    AdamConfig | AdamWConfig | SGDConfig,
    C.Field(discriminator="name"),
]


def create_optimizer(
    config: OptimizerConfig,
    parameters: Iterable[nn.Parameter],
) -> torch.optim.Optimizer:
    match config:
        case AdamConfig():
            optimizer = torch.optim.Adam(
                parameters,
                lr=config.lr,
                eps=config.eps,
                betas=config.betas,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad,
            )
        case AdamWConfig():
            optimizer = torch.optim.AdamW(
                parameters,
                lr=config.lr,
                eps=config.eps,
                betas=config.betas,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad,
            )
        case SGDConfig():
            optimizer = torch.optim.SGD(
                parameters,
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nestrov,
            )
        case _:
            assert_never(optimizer)

    return optimizer
