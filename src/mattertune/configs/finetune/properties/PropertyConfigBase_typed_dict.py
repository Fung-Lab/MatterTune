from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.properties import PropertyConfigBase


__codegen__ = True

# Definitions


class HuberLossConfig(typ.TypedDict, total=False):
    name: typ.Literal["huber"]

    delta: float
    """The threshold value for the Huber loss function."""

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


class L2MAELossConfig(typ.TypedDict, total=False):
    name: typ.Literal["l2_mae"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


class MAELossConfig(typ.TypedDict, total=False):
    name: typ.Literal["mae"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


class MSELossConfig(typ.TypedDict, total=False):
    name: typ.Literal["mse"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


# Schema entries
class PropertyConfigBaseTypedDict(typ.TypedDict):
    name: str
    """The name of the property.
    This is the key that will be used to access the property in the output of the model.
    This is also the key that will be used to access the property in the ASE Atoms object."""

    dtype: typ.Literal["float"]
    """The type of the property values."""

    loss: MAELossConfig | MSELossConfig | HuberLossConfig | L2MAELossConfig
    """The loss function to use when training the model on this property."""

    loss_coefficient: typ.NotRequired[float]
    """The coefficient to apply to this property's loss function when training the model."""


@typ.overload
def CreatePropertyConfigBase(
    dict: PropertyConfigBaseTypedDict, /
) -> PropertyConfigBase: ...


@typ.overload
def CreatePropertyConfigBase(
    **dict: typ.Unpack[PropertyConfigBaseTypedDict],
) -> PropertyConfigBase: ...


def CreatePropertyConfigBase(*args, **kwargs):
    from mattertune.finetune.properties import PropertyConfigBase

    dict = args[0] if args else kwargs
    return PropertyConfigBase.model_validate(dict)
