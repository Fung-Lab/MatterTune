from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.normalization import NormalizerConfigBase


__codegen__ = True


# Schema entries
class NormalizerConfigBaseTypedDict(typ.TypedDict, total=False):
    pass


@typ.overload
def CreateNormalizerConfigBase(
    dict: NormalizerConfigBaseTypedDict, /
) -> NormalizerConfigBase: ...


@typ.overload
def CreateNormalizerConfigBase(
    **dict: typ.Unpack[NormalizerConfigBaseTypedDict],
) -> NormalizerConfigBase: ...


def CreateNormalizerConfigBase(*args, **kwargs):
    from mattertune.normalization import NormalizerConfigBase

    dict = args[0] if args else kwargs
    return NormalizerConfigBase.model_validate(dict)
