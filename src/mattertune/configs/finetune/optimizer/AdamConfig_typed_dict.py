from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.optimizer import AdamConfig


__codegen__ = True


# Schema entries
class AdamConfigTypedDict(typ.TypedDict, total=False):
    name: typ.Literal["Adam"]
    """Name of the optimizer."""

    lr: typ.Required[float]
    """Learning rate."""

    eps: float
    """Epsilon."""

    betas: tuple[float, float]
    """Betas."""

    weight_decay: float
    """Weight decay."""

    amsgrad: bool
    """Whether to use AMSGrad variant of Adam."""


@typ.overload
def CreateAdamConfig(dict: AdamConfigTypedDict, /) -> AdamConfig: ...


@typ.overload
def CreateAdamConfig(**dict: typ.Unpack[AdamConfigTypedDict]) -> AdamConfig: ...


def CreateAdamConfig(*args, **kwargs):
    from mattertune.finetune.optimizer import AdamConfig

    dict = args[0] if args else kwargs
    return AdamConfig.model_validate(dict)
