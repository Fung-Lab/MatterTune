from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.mptraj import MPTrajDatasetConfig


__codegen__ = True

"""Configuration for a dataset stored in the Materials Project database."""


# Schema entries
class MPTrajDatasetConfigTypedDict(typ.TypedDict, total=False):
    """Configuration for a dataset stored in the Materials Project database."""

    type: typ.Literal["mptraj"]
    """Discriminator for the MPTraj dataset."""

    split: typ.Literal["train"] | typ.Literal["val"] | typ.Literal["test"]
    """Split of the dataset to use."""

    min_num_atoms: int | None
    """Minimum number of atoms to be considered. Drops structures with fewer atoms."""

    max_num_atoms: int | None
    """Maximum number of atoms to be considered. Drops structures with more atoms."""

    elements: list[str] | None
    """List of elements to be considered. Drops structures with elements not in the list.
    Subsets are also allowed. For example, ["Li", "Na"] will keep structures with either Li or Na."""


@typ.overload
def CreateMPTrajDatasetConfig(
    dict: MPTrajDatasetConfigTypedDict, /
) -> MPTrajDatasetConfig: ...


@typ.overload
def CreateMPTrajDatasetConfig(
    **dict: typ.Unpack[MPTrajDatasetConfigTypedDict],
) -> MPTrajDatasetConfig: ...


def CreateMPTrajDatasetConfig(*args, **kwargs):
    from mattertune.data.mptraj import MPTrajDatasetConfig

    dict = args[0] if args else kwargs
    return MPTrajDatasetConfig.model_validate(dict)
