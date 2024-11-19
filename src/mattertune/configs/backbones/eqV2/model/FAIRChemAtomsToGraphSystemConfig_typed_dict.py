from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.backbones.eqV2.model import FAIRChemAtomsToGraphSystemConfig


__codegen__ = True

"""Configuration for converting ASE Atoms to a graph for the FAIRChem model."""


# Schema entries
class FAIRChemAtomsToGraphSystemConfigTypedDict(typ.TypedDict):
    """Configuration for converting ASE Atoms to a graph for the FAIRChem model."""

    radius: float
    """The radius for edge construction."""

    max_num_neighbors: int
    """The maximum number of neighbours each node can send messages to."""


@typ.overload
def CreateFAIRChemAtomsToGraphSystemConfig(
    dict: FAIRChemAtomsToGraphSystemConfigTypedDict, /
) -> FAIRChemAtomsToGraphSystemConfig: ...


@typ.overload
def CreateFAIRChemAtomsToGraphSystemConfig(
    **dict: typ.Unpack[FAIRChemAtomsToGraphSystemConfigTypedDict],
) -> FAIRChemAtomsToGraphSystemConfig: ...


def CreateFAIRChemAtomsToGraphSystemConfig(*args, **kwargs):
    from mattertune.backbones.eqV2.model import FAIRChemAtomsToGraphSystemConfig

    dict = args[0] if args else kwargs
    return FAIRChemAtomsToGraphSystemConfig.model_validate(dict)
