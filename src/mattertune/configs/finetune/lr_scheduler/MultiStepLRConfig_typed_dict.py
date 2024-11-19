from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.finetune.lr_scheduler import MultiStepLRConfig


__codegen__ = True


# Schema entries
class MultiStepLRConfigTypedDict(typ.TypedDict):
    type: typ.NotRequired[typ.Literal["MultiStepLR"]]
    """Type of the learning rate scheduler."""

    milestones: list[int]
    """List of epoch indices. Must be increasing."""

    gamma: float
    """Multiplicative factor of learning rate decay."""


@typ.overload
def CreateMultiStepLRConfig(
    dict: MultiStepLRConfigTypedDict, /
) -> MultiStepLRConfig: ...


@typ.overload
def CreateMultiStepLRConfig(
    **dict: typ.Unpack[MultiStepLRConfigTypedDict],
) -> MultiStepLRConfig: ...


def CreateMultiStepLRConfig(*args, **kwargs):
    from mattertune.finetune.lr_scheduler import MultiStepLRConfig

    dict = args[0] if args else kwargs
    return MultiStepLRConfig.model_validate(dict)
