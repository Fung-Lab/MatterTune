from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.backbones.jmp.model import JMPGraphComputerConfig


__codegen__ = True

# Definitions


class CutoffsConfig(typ.TypedDict):
    main: float

    aeaint: float

    qint: float

    aint: float


class MaxNeighborsConfig(typ.TypedDict):
    main: int

    aeaint: int

    qint: int

    aint: int


# Schema entries
class JMPGraphComputerConfigTypedDict(typ.TypedDict, total=False):
    pbc: typ.Required[bool]
    """Whether to use periodic boundary conditions."""

    cutoffs: CutoffsConfig
    """The cutoff for the radius graph."""

    max_neighbors: MaxNeighborsConfig
    """The maximum number of neighbors for the radius graph."""

    per_graph_radius_graph: bool
    """Whether to compute the radius graph per graph."""


@typ.overload
def CreateJMPGraphComputerConfig(
    dict: JMPGraphComputerConfigTypedDict, /
) -> JMPGraphComputerConfig: ...


@typ.overload
def CreateJMPGraphComputerConfig(
    **dict: typ.Unpack[JMPGraphComputerConfigTypedDict],
) -> JMPGraphComputerConfig: ...


def CreateJMPGraphComputerConfig(*args, **kwargs):
    from mattertune.backbones.jmp.model import JMPGraphComputerConfig

    dict = args[0] if args else kwargs
    return JMPGraphComputerConfig.model_validate(dict)
