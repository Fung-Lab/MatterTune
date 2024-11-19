from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.optimizer import AdamWConfig


__codegen__ = True


# Schema entries
class AdamWConfigTypedDict(typ.TypedDict, total=False):
    name: typ.Literal["AdamW"]
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
def CreateAdamWConfig(dict: AdamWConfigTypedDict, /) -> AdamWConfig: ...


@typ.overload
def CreateAdamWConfig(**dict: typ.Unpack[AdamWConfigTypedDict]) -> AdamWConfig: ...


def CreateAdamWConfig(*args, **kwargs):
    from mattertune.finetune.optimizer import AdamWConfig

    dict = args[0] if args else kwargs
    return AdamWConfig.model_validate(dict)
