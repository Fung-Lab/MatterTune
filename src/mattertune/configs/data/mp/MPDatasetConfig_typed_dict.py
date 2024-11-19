from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.mp import MPDatasetConfig


__codegen__ = True

"""Configuration for a dataset stored in the Materials Project database."""


# Schema entries
class MPDatasetConfigTypedDictQuery(typ.TypedDict, total=False):
    """Query to filter the data from the Materials Project database."""

    pass


class MPDatasetConfigTypedDict(typ.TypedDict):
    """Configuration for a dataset stored in the Materials Project database."""

    type: typ.NotRequired[typ.Literal["mp"]]
    """Discriminator for the MP dataset."""

    api: str
    """Input API key for the Materials Project database."""

    fields: list[str]
    """Fields to retrieve from the Materials Project database."""

    query: MPDatasetConfigTypedDictQuery
    """Query to filter the data from the Materials Project database."""


@typ.overload
def CreateMPDatasetConfig(dict: MPDatasetConfigTypedDict, /) -> MPDatasetConfig: ...


@typ.overload
def CreateMPDatasetConfig(
    **dict: typ.Unpack[MPDatasetConfigTypedDict],
) -> MPDatasetConfig: ...


def CreateMPDatasetConfig(*args, **kwargs):
    from mattertune.data.mp import MPDatasetConfig

    dict = args[0] if args else kwargs
    return MPDatasetConfig.model_validate(dict)
