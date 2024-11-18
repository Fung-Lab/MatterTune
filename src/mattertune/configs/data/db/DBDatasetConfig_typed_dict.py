from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.db import DBDatasetConfig


__codegen__ = True

"""Configuration for a dataset stored in an ASE database."""


# Schema entries
class DBDatasetConfigTypedDict(typ.TypedDict, total=False):
    """Configuration for a dataset stored in an ASE database."""

    type: typ.Literal["db"]
    """Discriminator for the DB dataset."""

    src: typ.Required[str | str]
    """Path to the ASE database file or a database object."""

    energy_key: str | None
    """Key for the energy label in the database."""

    forces_key: str | None
    """Key for the force label in the database."""

    stress_key: str | None
    """Key for the stress label in the database."""

    preload: bool
    """Whether to load all the data at once or not."""


@typ.overload
def CreateDBDatasetConfig(dict: DBDatasetConfigTypedDict, /) -> DBDatasetConfig: ...


@typ.overload
def CreateDBDatasetConfig(
    **dict: typ.Unpack[DBDatasetConfigTypedDict],
) -> DBDatasetConfig: ...


def CreateDBDatasetConfig(*args, **kwargs):
    from mattertune.data.db import DBDatasetConfig

    dict = args[0] if args else kwargs
    return DBDatasetConfig.model_validate(dict)
