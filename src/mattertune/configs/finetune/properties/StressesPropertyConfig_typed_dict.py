from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.properties import StressesPropertyConfig


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
class StressesPropertyConfigTypedDict(typ.TypedDict, total=False):
    name: str
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""

    dtype: typ.Literal["float"]
    """The type of the property values."""

    loss: typ.Required[
        MAELossConfig | MSELossConfig | HuberLossConfig | L2MAELossConfig
    ]
    """The loss function to use when training the model on this property."""

    loss_coefficient: float
    """The coefficient to apply to this property's loss function when training the model."""

    type: typ.Literal["stresses"]

    conservative: typ.Required[bool]
    """Similar to the `conservative` parameter in `ForcesPropertyConfig`, this parameter
        specifies whether the stresses should be computed in a conservative manner."""


@typ.overload
def CreateStressesPropertyConfig(
    dict: StressesPropertyConfigTypedDict, /
) -> StressesPropertyConfig: ...


@typ.overload
def CreateStressesPropertyConfig(
    **dict: typ.Unpack[StressesPropertyConfigTypedDict],
) -> StressesPropertyConfig: ...


def CreateStressesPropertyConfig(*args, **kwargs):
    from mattertune.finetune.properties import StressesPropertyConfig

    dict = args[0] if args else kwargs
    return StressesPropertyConfig.model_validate(dict)
