from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.normalization import RMSNormalizerConfig


__codegen__ = True


# Schema entries
class RMSNormalizerConfigTypedDict(typ.TypedDict):
    rms: float
    """The root mean square of the property values."""


@typ.overload
def CreateRMSNormalizerConfig(
    dict: RMSNormalizerConfigTypedDict, /
) -> RMSNormalizerConfig: ...


@typ.overload
def CreateRMSNormalizerConfig(
    **dict: typ.Unpack[RMSNormalizerConfigTypedDict],
) -> RMSNormalizerConfig: ...


def CreateRMSNormalizerConfig(*args, **kwargs):
    from mattertune.normalization import RMSNormalizerConfig

    dict = args[0] if args else kwargs
    return RMSNormalizerConfig.model_validate(dict)
