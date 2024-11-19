from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.loss import MSELossConfig


__codegen__ = True


# Schema entries
class MSELossConfigTypedDict(typ.TypedDict, total=False):
    name: typ.Literal["mse"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


@typ.overload
def CreateMSELossConfig(dict: MSELossConfigTypedDict, /) -> MSELossConfig: ...


@typ.overload
def CreateMSELossConfig(
    **dict: typ.Unpack[MSELossConfigTypedDict],
) -> MSELossConfig: ...


def CreateMSELossConfig(*args, **kwargs):
    from mattertune.finetune.loss import MSELossConfig

    dict = args[0] if args else kwargs
    return MSELossConfig.model_validate(dict)
