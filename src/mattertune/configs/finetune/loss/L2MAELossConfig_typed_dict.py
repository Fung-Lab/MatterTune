from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.loss import L2MAELossConfig


__codegen__ = True


# Schema entries
class L2MAELossConfigTypedDict(typ.TypedDict, total=False):
    name: typ.Literal["l2_mae"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


@typ.overload
def CreateL2MAELossConfig(dict: L2MAELossConfigTypedDict, /) -> L2MAELossConfig: ...


@typ.overload
def CreateL2MAELossConfig(
    **dict: typ.Unpack[L2MAELossConfigTypedDict],
) -> L2MAELossConfig: ...


def CreateL2MAELossConfig(*args, **kwargs):
    from mattertune.finetune.loss import L2MAELossConfig

    dict = args[0] if args else kwargs
    return L2MAELossConfig.model_validate(dict)
