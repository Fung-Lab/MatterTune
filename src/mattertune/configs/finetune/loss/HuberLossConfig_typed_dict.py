from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.loss import HuberLossConfig


__codegen__ = True


# Schema entries
class HuberLossConfigTypedDict(typ.TypedDict, total=False):
    name: typ.Literal["huber"]

    delta: float
    """The threshold value for the Huber loss function."""

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


@typ.overload
def CreateHuberLossConfig(dict: HuberLossConfigTypedDict, /) -> HuberLossConfig: ...


@typ.overload
def CreateHuberLossConfig(
    **dict: typ.Unpack[HuberLossConfigTypedDict],
) -> HuberLossConfig: ...


def CreateHuberLossConfig(*args, **kwargs):
    from mattertune.finetune.loss import HuberLossConfig

    dict = args[0] if args else kwargs
    return HuberLossConfig.model_validate(dict)
