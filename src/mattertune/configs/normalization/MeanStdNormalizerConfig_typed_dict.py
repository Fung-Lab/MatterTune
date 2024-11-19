from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.normalization import MeanStdNormalizerConfig


__codegen__ = True


# Schema entries
class MeanStdNormalizerConfigTypedDict(typ.TypedDict):
    mean: float
    """The mean of the property values."""

    std: float
    """The standard deviation of the property values."""


@typ.overload
def CreateMeanStdNormalizerConfig(
    dict: MeanStdNormalizerConfigTypedDict, /
) -> MeanStdNormalizerConfig: ...


@typ.overload
def CreateMeanStdNormalizerConfig(
    **dict: typ.Unpack[MeanStdNormalizerConfigTypedDict],
) -> MeanStdNormalizerConfig: ...


def CreateMeanStdNormalizerConfig(*args, **kwargs):
    from mattertune.normalization import MeanStdNormalizerConfig

    dict = args[0] if args else kwargs
    return MeanStdNormalizerConfig.model_validate(dict)
