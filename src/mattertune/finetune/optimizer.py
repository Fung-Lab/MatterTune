from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Sequence
from typing import Annotated, Any, Literal

import nshconfig as C
import torch
import torch.nn as nn
from typing_extensions import NotRequired, TypeAliasType, TypedDict, assert_never


class PerParamHparamsDict(TypedDict):
    patterns: Sequence[str]
    """Patterns to match parameter names."""

    hparams: dict[str, Any]
    """Hyperparameters for the matched parameters."""

    optimize: NotRequired[bool]
    """Whether to optimize this parameter. Default is True."""


class OptimizerConfigBase(C.Config):
    per_parameter_hparams: Sequence[PerParamHparamsDict] | None = None
    """Per parameter hyperparameters.

    This should be a list of dictionaries, each of which has the following keys:

    - `patterns`: a list of patterns to match parameter names.
    - `hparams`: a dictionary of hyperparameters for the matched parameters.
    - `optimize`: whether to optimize this parameter. Default is True.

    This allows you to, for example, set different learning rates for
    different parameters."""


class AdamConfig(OptimizerConfigBase):
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


class AdamWConfig(OptimizerConfigBase):
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


class SGDConfig(OptimizerConfigBase):
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


OptimizerConfig = TypeAliasType(
    "OptimizerConfig",
    Annotated[
        AdamConfig | AdamWConfig | SGDConfig,
        C.Field(discriminator="name"),
    ],
)


def _named_parameters_matching_patterns(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    patterns: Iterable[str],
):
    for name, param in named_parameters:
        if (
            matching_pattern := next(
                (pattern for pattern in patterns if fnmatch.fnmatch(name, pattern)),
                None,
            )
        ) is None:
            continue

        yield name, param, matching_pattern


def     _split_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    pattern_lists: Iterable[Iterable[str]],
):
    named_parameters_list = list(named_parameters)
    all_parameters = [p for _, p in named_parameters_list]

    parameters: list[list[torch.nn.Parameter]] = []
    for patterns in pattern_lists:
        matching = [
            p
            for _, p, _ in _named_parameters_matching_patterns(
                named_parameters_list, patterns
            )
        ]
        parameters.append(matching)

        # Remove matching parameters from all_parameters
        all_parameters = [
            p for p in all_parameters if all(p is not m for m in matching)
        ]

    return parameters, all_parameters


def create_optimizer(
    config: OptimizerConfig,
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> torch.optim.Optimizer:
    default_kwargs: dict[str, Any]
    match config:
        case AdamConfig():
            default_kwargs = dict(
                lr=config.lr,
                eps=config.eps,
                betas=config.betas,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad,
            )
            cls = torch.optim.Adam
        case AdamWConfig():
            default_kwargs = dict(
                lr=config.lr,
                eps=config.eps,
                betas=config.betas,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad,
            )
            cls = torch.optim.AdamW
        case SGDConfig():
            default_kwargs = dict(
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nestrov,
            )
            cls = torch.optim.SGD
        case _:
            assert_never(config)

    # If per_parameter_hparams is not specified, return the optimizer
    if config.per_parameter_hparams is None:
        return cls((p for _, p in named_parameters), **default_kwargs)

    # Otherwise, split parameters
    parameters, all_parameters = _split_parameters(
        named_parameters, [d["patterns"] for d in config.per_parameter_hparams]
    )

    params_list: list[dict[str, Any]] = []
    for p, d in zip(parameters, config.per_parameter_hparams):
        if not d.get("optimize", True):
            continue

        param_dict = {}
        param_dict.update(default_kwargs)
        param_dict.update(d["hparams"])
        param_dict["params"] = p
        params_list.append(param_dict)

    if all_parameters:
        params_list.append({"params": all_parameters, **default_kwargs})

    return cls(params_list, **default_kwargs)
