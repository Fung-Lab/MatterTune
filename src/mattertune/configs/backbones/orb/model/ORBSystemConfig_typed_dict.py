from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.backbones.orb.model import ORBSystemConfig


__codegen__ = True

"""Config controlling how to featurize a system of atoms."""


# Schema entries
class ORBSystemConfigTypedDict(typ.TypedDict):
    """Config controlling how to featurize a system of atoms."""

    radius: float
    """The radius for edge construction."""

    max_num_neighbors: int
    """The maximum number of neighbours each node can send messages to."""


@typ.overload
def CreateORBSystemConfig(dict: ORBSystemConfigTypedDict, /) -> ORBSystemConfig: ...


@typ.overload
def CreateORBSystemConfig(
    **dict: typ.Unpack[ORBSystemConfigTypedDict],
) -> ORBSystemConfig: ...


def CreateORBSystemConfig(*args, **kwargs):
    from mattertune.backbones.orb.model import ORBSystemConfig

    dict = args[0] if args else kwargs
    return ORBSystemConfig.model_validate(dict)
