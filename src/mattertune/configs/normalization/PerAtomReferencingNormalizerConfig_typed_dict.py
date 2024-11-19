from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.normalization import PerAtomReferencingNormalizerConfig


__codegen__ = True


# Schema entries
class PerAtomReferencingNormalizerConfigTypedDict(typ.TypedDict):
    per_atom_references: dict[str, float] | list[float] | str
    """The reference values for each element.
    
    - If a dictionary is provided, it maps atomic numbers to reference values
    - If a list is provided, it's a list of reference values indexed by atomic number
    - If a path is provided, it should point to a JSON file containing the references."""


@typ.overload
def CreatePerAtomReferencingNormalizerConfig(
    dict: PerAtomReferencingNormalizerConfigTypedDict, /
) -> PerAtomReferencingNormalizerConfig: ...


@typ.overload
def CreatePerAtomReferencingNormalizerConfig(
    **dict: typ.Unpack[PerAtomReferencingNormalizerConfigTypedDict],
) -> PerAtomReferencingNormalizerConfig: ...


def CreatePerAtomReferencingNormalizerConfig(*args, **kwargs):
    from mattertune.normalization import PerAtomReferencingNormalizerConfig

    dict = args[0] if args else kwargs
    return PerAtomReferencingNormalizerConfig.model_validate(dict)
