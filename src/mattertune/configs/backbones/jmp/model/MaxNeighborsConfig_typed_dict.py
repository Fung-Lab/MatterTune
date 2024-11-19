from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.backbones.jmp.model import MaxNeighborsConfig


__codegen__ = True


# Schema entries
class MaxNeighborsConfigTypedDict(typ.TypedDict):
    main: int

    aeaint: int

    qint: int

    aint: int


@typ.overload
def CreateMaxNeighborsConfig(
    dict: MaxNeighborsConfigTypedDict, /
) -> MaxNeighborsConfig: ...


@typ.overload
def CreateMaxNeighborsConfig(
    **dict: typ.Unpack[MaxNeighborsConfigTypedDict],
) -> MaxNeighborsConfig: ...


def CreateMaxNeighborsConfig(*args, **kwargs):
    from mattertune.backbones.jmp.model import MaxNeighborsConfig

    dict = args[0] if args else kwargs
    return MaxNeighborsConfig.model_validate(dict)
