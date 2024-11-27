from __future__ import annotations

from typing import Annotated, Literal

import nshconfig as C
import torch
from typing_extensions import TypeAliasType, assert_never


class StepLRConfig(C.Config):
    type: Literal["StepLR"] = "StepLR"
    """Type of the learning rate scheduler."""
    step_size: int
    """Period of learning rate decay."""
    gamma: float
    """Multiplicative factor of learning rate decay."""


class MultiStepLRConfig(C.Config):
    type: Literal["MultiStepLR"] = "MultiStepLR"
    """Type of the learning rate scheduler."""
    milestones: list[int]
    """List of epoch indices. Must be increasing."""
    gamma: float
    """Multiplicative factor of learning rate decay."""


class ExponentialConfig(C.Config):
    type: Literal["ExponentialLR"] = "ExponentialLR"
    """Type of the learning rate scheduler."""
    gamma: float
    """Multiplicative factor of learning rate decay."""


class ReduceOnPlateauConfig(C.Config):
    type: Literal["ReduceLROnPlateau"] = "ReduceLROnPlateau"
    """Type of the learning rate scheduler."""
    mode: str
    """One of {"min", "max"}. Determines when to reduce the learning rate."""
    factor: float
    """Factor by which the learning rate will be reduced."""
    patience: int
    """Number of epochs with no improvement after which learning rate will be reduced."""
    threshold: float = 1e-4
    """Threshold for measuring the new optimum."""
    threshold_mode: str = "rel"
    """One of {"rel", "abs"}. Determines the threshold mode."""
    cooldown: int = 0
    """Number of epochs to wait before resuming normal operation."""
    min_lr: float = 0
    """A lower bound on the learning rate."""
    eps: float = 1e-8
    """Threshold for testing the new optimum."""


class CosineAnnealingLRConfig(C.Config):
    type: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"
    """Type of the learning rate scheduler."""
    T_max: int
    """Maximum number of iterations."""
    eta_min: float = 0
    """Minimum learning rate."""
    last_epoch: int = -1
    """The index of last epoch."""


LRSchedulerConfig = TypeAliasType(
    "LRSchedulerConfig",
    Annotated[
        StepLRConfig
        | MultiStepLRConfig
        | ExponentialConfig
        | ReduceOnPlateauConfig
        | CosineAnnealingLRConfig,
        C.Field(discriminator="type"),
    ],
)


def create_lr_scheduler(
    config: LRSchedulerConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    match config:
        case StepLRConfig():
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma,
            )
        case MultiStepLRConfig():
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.milestones,
                gamma=config.gamma,
            )
        case ExponentialConfig():
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.gamma,
            )
        case ReduceOnPlateauConfig():
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.mode,
                factor=config.factor,
                patience=config.patience,
                threshold=config.threshold,
                threshold_mode=config.threshold_mode,
                cooldown=config.cooldown,
                min_lr=config.min_lr,
                eps=config.eps,
            )
        case CosineAnnealingLRConfig():
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.T_max,
                eta_min=config.eta_min,
                last_epoch=config.last_epoch,
            )
        case _:
            assert_never(config)

    return lr_scheduler
