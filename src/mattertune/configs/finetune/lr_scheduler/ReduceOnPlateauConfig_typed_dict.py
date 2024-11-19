from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.lr_scheduler import ReduceOnPlateauConfig


__codegen__ = True


# Schema entries
class ReduceOnPlateauConfigTypedDict(typ.TypedDict, total=False):
    type: typ.Literal["ReduceLROnPlateau"]
    """Type of the learning rate scheduler."""

    mode: typ.Required[str]
    """One of {"min", "max"}. Determines when to reduce the learning rate."""

    factor: typ.Required[float]
    """Factor by which the learning rate will be reduced."""

    patience: typ.Required[int]
    """Number of epochs with no improvement after which learning rate will be reduced."""

    threshold: float
    """Threshold for measuring the new optimum."""

    threshold_mode: str
    """One of {"rel", "abs"}. Determines the threshold mode."""

    cooldown: int
    """Number of epochs to wait before resuming normal operation."""

    min_lr: float
    """A lower bound on the learning rate."""

    eps: float
    """Threshold for testing the new optimum."""


@typ.overload
def CreateReduceOnPlateauConfig(
    dict: ReduceOnPlateauConfigTypedDict, /
) -> ReduceOnPlateauConfig: ...


@typ.overload
def CreateReduceOnPlateauConfig(
    **dict: typ.Unpack[ReduceOnPlateauConfigTypedDict],
) -> ReduceOnPlateauConfig: ...


def CreateReduceOnPlateauConfig(*args, **kwargs):
    from mattertune.finetune.lr_scheduler import ReduceOnPlateauConfig

    dict = args[0] if args else kwargs
    return ReduceOnPlateauConfig.model_validate(dict)
