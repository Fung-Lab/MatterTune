from __future__ import annotations

from typing import Annotated, Literal

import nshconfig as C
import torch
import torch.optim.lr_scheduler as lrs
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
    mode: Literal["min", "max"]
    """One of {"min", "max"}. Determines when to reduce the learning rate."""
    monitor: str = "val_loss"
    """Quantity to be monitored."""
    factor: float
    """Factor by which the learning rate will be reduced."""
    patience: int
    """Number of epochs with no improvement after which learning rate will be reduced."""
    threshold: float = 1e-4
    """Threshold for measuring the new optimum."""
    threshold_mode: Literal["rel", "abs"] = "rel"
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


class ConstantLRConfig(C.Config):
    type: Literal["ConstantLR"] = "ConstantLR"
    """Type of the learning rate scheduler."""
    factor: float = 1.0 / 3
    """The number we multiply learning rate until the milestone."""
    total_iters: int = 5
    """The number of steps that the scheduler decays the learning rate."""


class LinearLRConfig(C.Config):
    type: Literal["LinearLR"] = "LinearLR"
    """Type of the learning rate scheduler."""
    start_factor: float = 1.0 / 3
    """The number we multiply learning rate in the first epoch."""
    end_factor: float = 1.0
    """The number we multiply learning rate at the end of linear changing process."""
    total_iters: int = 5
    """The number of iterations that multiplicative factor reaches to 1."""


SingleLRSchedulerConfig = TypeAliasType(
    "SingleLRSchedulerConfig",
    Annotated[
        StepLRConfig
        | MultiStepLRConfig
        | ExponentialConfig
        | ReduceOnPlateauConfig
        | CosineAnnealingLRConfig
        | ConstantLRConfig
        | LinearLRConfig,
        C.Field(discriminator="type"),
    ],
)


LRSchedulerConfig = TypeAliasType(
    "LRSchedulerConfig",
    SingleLRSchedulerConfig | list[SingleLRSchedulerConfig],
)


def create_single_lr_scheduler(
    config: SingleLRSchedulerConfig,
    optimizer: torch.optim.Optimizer,
) -> lrs.LRScheduler:
    match config:
        case StepLRConfig():
            lr_scheduler = lrs.StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma,
            )
        case MultiStepLRConfig():
            lr_scheduler = lrs.MultiStepLR(
                optimizer,
                milestones=config.milestones,
                gamma=config.gamma,
            )
        case ExponentialConfig():
            lr_scheduler = lrs.ExponentialLR(
                optimizer,
                gamma=config.gamma,
            )
        case ReduceOnPlateauConfig():
            lr_scheduler = lrs.ReduceLROnPlateau(
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
            lr_scheduler = lrs.CosineAnnealingLR(
                optimizer,
                T_max=config.T_max,
                eta_min=config.eta_min,
                last_epoch=config.last_epoch,
            )
        case ConstantLRConfig():
            lr_scheduler = lrs.ConstantLR(
                optimizer,
                factor=config.factor,
                total_iters=config.total_iters,
            )
        case LinearLRConfig():
            lr_scheduler = lrs.LinearLR(
                optimizer,
                start_factor=config.start_factor,
                end_factor=config.end_factor,
                total_iters=config.total_iters,
            )
        case _:
            assert_never(config)

    return lr_scheduler


def create_lr_scheduler(
    config: LRSchedulerConfig,
    optimizer: torch.optim.Optimizer,
) -> lrs.LRScheduler:
    if isinstance(config, list) and len(config) == 1:
        config = config[0]

    if isinstance(config, list):
        # Throw an error if the list is empty.
        if not config:
            raise ValueError("The list of learning rate schedulers is empty.")

        # Throw an error if the list contains ReduceOnPlateauConfig,
        # which is not supported by ChainedScheduler.
        if any(isinstance(cfg, ReduceOnPlateauConfig) for cfg in config):
            raise ValueError(
                "ReduceOnPlateauConfig is not supported when using multiple learning rate schedulers."
            )

        lr_schedulers = [
            create_single_lr_scheduler(single_config, optimizer)
            for single_config in config
        ]
        return lrs.ChainedScheduler(lr_schedulers)

    return create_single_lr_scheduler(config, optimizer)
