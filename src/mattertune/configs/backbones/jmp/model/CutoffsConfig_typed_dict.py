from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.backbones.jmp.model import CutoffsConfig


__codegen__ = True


# Schema entries
class CutoffsConfigTypedDict(typ.TypedDict):
    main: float

    aeaint: float

    qint: float

    aint: float


@typ.overload
def CreateCutoffsConfig(dict: CutoffsConfigTypedDict, /) -> CutoffsConfig: ...


@typ.overload
def CreateCutoffsConfig(
    **dict: typ.Unpack[CutoffsConfigTypedDict],
) -> CutoffsConfig: ...


def CreateCutoffsConfig(*args, **kwargs):
    from mattertune.backbones.jmp.model import CutoffsConfig

    dict = args[0] if args else kwargs
    return CutoffsConfig.model_validate(dict)
