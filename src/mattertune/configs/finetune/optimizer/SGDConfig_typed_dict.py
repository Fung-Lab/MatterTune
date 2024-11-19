from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.optimizer import SGDConfig


__codegen__ = True


# Schema entries
class SGDConfigTypedDict(typ.TypedDict, total=False):
    name: typ.Literal["SGD"]
    """Name of the optimizer."""

    lr: typ.Required[float]
    """Learning rate."""

    momentum: float
    """Momentum."""

    weight_decay: float
    """Weight decay."""

    nestrov: bool
    """Whether to use nestrov."""


@typ.overload
def CreateSGDConfig(dict: SGDConfigTypedDict, /) -> SGDConfig: ...


@typ.overload
def CreateSGDConfig(**dict: typ.Unpack[SGDConfigTypedDict]) -> SGDConfig: ...


def CreateSGDConfig(*args, **kwargs):
    from mattertune.finetune.optimizer import SGDConfig

    dict = args[0] if args else kwargs
    return SGDConfig.model_validate(dict)
