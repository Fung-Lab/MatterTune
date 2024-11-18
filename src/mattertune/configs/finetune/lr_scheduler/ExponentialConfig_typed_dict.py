from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.lr_scheduler import ExponentialConfig


__codegen__ = True


# Schema entries
class ExponentialConfigTypedDict(typ.TypedDict, total=False):
    type: typ.Literal["ExponentialLR"]
    """Type of the learning rate scheduler."""

    gamma: typ.Required[float]
    """Multiplicative factor of learning rate decay."""


@typ.overload
def CreateExponentialConfig(
    dict: ExponentialConfigTypedDict, /
) -> ExponentialConfig: ...


@typ.overload
def CreateExponentialConfig(
    **dict: typ.Unpack[ExponentialConfigTypedDict],
) -> ExponentialConfig: ...


def CreateExponentialConfig(*args, **kwargs):
    from mattertune.finetune.lr_scheduler import ExponentialConfig

    dict = args[0] if args else kwargs
    return ExponentialConfig.model_validate(dict)
