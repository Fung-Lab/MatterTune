from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.lr_scheduler import StepLRConfig


__codegen__ = True


# Schema entries
class StepLRConfigTypedDict(typ.TypedDict):
    type: typ.NotRequired[typ.Literal["StepLR"]]
    """Type of the learning rate scheduler."""

    step_size: int
    """Period of learning rate decay."""

    gamma: float
    """Multiplicative factor of learning rate decay."""


@typ.overload
def CreateStepLRConfig(dict: StepLRConfigTypedDict, /) -> StepLRConfig: ...


@typ.overload
def CreateStepLRConfig(**dict: typ.Unpack[StepLRConfigTypedDict]) -> StepLRConfig: ...


def CreateStepLRConfig(*args, **kwargs):
    from mattertune.finetune.lr_scheduler import StepLRConfig

    dict = args[0] if args else kwargs
    return StepLRConfig.model_validate(dict)
