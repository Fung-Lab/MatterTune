from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.backbones.m3gnet.model import M3GNetGraphComputerConfig


__codegen__ = True

"""Configuration for initialize a MatGL Atoms2Graph Convertor."""


# Schema entries
class M3GNetGraphComputerConfigTypedDict(typ.TypedDict, total=False):
    """Configuration for initialize a MatGL Atoms2Graph Convertor."""

    element_types: list[str]
    """The element types to consider, default is all elements."""

    cutoff: float | None
    """The cutoff distance for the neighbor list. If None, the cutoff is loaded from the checkpoint."""

    threebody_cutoff: float | None
    """The cutoff distance for the three-body interactions. If None, the cutoff is loaded from the checkpoint."""

    pre_compute_line_graph: bool
    """Whether to pre-compute the line graph for three-body interactions in data preparation."""

    graph_labels: list[int | float] | None
    """The graph labels to consider, default is None."""


@typ.overload
def CreateM3GNetGraphComputerConfig(
    dict: M3GNetGraphComputerConfigTypedDict, /
) -> M3GNetGraphComputerConfig: ...


@typ.overload
def CreateM3GNetGraphComputerConfig(
    **dict: typ.Unpack[M3GNetGraphComputerConfigTypedDict],
) -> M3GNetGraphComputerConfig: ...


def CreateM3GNetGraphComputerConfig(*args, **kwargs):
    from mattertune.backbones.m3gnet.model import M3GNetGraphComputerConfig

    dict = args[0] if args else kwargs
    return M3GNetGraphComputerConfig.model_validate(dict)
