from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.backbones.jmp.prediction_heads.graph_scalar import (
        GraphScalarTargetConfig,
    )


__codegen__ = True


# Schema entries
class GraphScalarTargetConfigTypedDict(typ.TypedDict, total=False):
    reduction: typ.Literal["mean"] | typ.Literal["sum"] | typ.Literal["max"]
    """The reduction to use for the output."""

    num_mlps: int
    """Number of MLPs in the output layer."""


@typ.overload
def CreateGraphScalarTargetConfig(
    dict: GraphScalarTargetConfigTypedDict, /
) -> GraphScalarTargetConfig: ...


@typ.overload
def CreateGraphScalarTargetConfig(
    **dict: typ.Unpack[GraphScalarTargetConfigTypedDict],
) -> GraphScalarTargetConfig: ...


def CreateGraphScalarTargetConfig(*args, **kwargs):
    from mattertune.backbones.jmp.prediction_heads.graph_scalar import (
        GraphScalarTargetConfig,
    )

    dict = args[0] if args else kwargs
    return GraphScalarTargetConfig.model_validate(dict)
