from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.loss import MAELossConfig


__codegen__ = True


# Schema entries
class MAELossConfigTypedDict(typ.TypedDict, total=False):
    name: typ.Literal["mae"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


@typ.overload
def CreateMAELossConfig(dict: MAELossConfigTypedDict, /) -> MAELossConfig: ...


@typ.overload
def CreateMAELossConfig(
    **dict: typ.Unpack[MAELossConfigTypedDict],
) -> MAELossConfig: ...


def CreateMAELossConfig(*args, **kwargs):
    from mattertune.finetune.loss import MAELossConfig

    dict = args[0] if args else kwargs
    return MAELossConfig.model_validate(dict)
