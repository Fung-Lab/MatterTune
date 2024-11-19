from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.lr_scheduler import CosineAnnealingLRConfig


__codegen__ = True


# Schema entries
class CosineAnnealingLRConfigTypedDict(typ.TypedDict, total=False):
    type: typ.Literal["CosineAnnealingLR"]
    """Type of the learning rate scheduler."""

    T_max: typ.Required[int]
    """Maximum number of iterations."""

    eta_min: float
    """Minimum learning rate."""

    last_epoch: int
    """The index of last epoch."""


@typ.overload
def CreateCosineAnnealingLRConfig(
    dict: CosineAnnealingLRConfigTypedDict, /
) -> CosineAnnealingLRConfig: ...


@typ.overload
def CreateCosineAnnealingLRConfig(
    **dict: typ.Unpack[CosineAnnealingLRConfigTypedDict],
) -> CosineAnnealingLRConfig: ...


def CreateCosineAnnealingLRConfig(*args, **kwargs):
    from mattertune.finetune.lr_scheduler import CosineAnnealingLRConfig

    dict = args[0] if args else kwargs
    return CosineAnnealingLRConfig.model_validate(dict)
