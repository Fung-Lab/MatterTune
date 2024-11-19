from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.omat24 import OMAT24DatasetConfig


__codegen__ = True


# Schema entries
class OMAT24DatasetConfigTypedDict(typ.TypedDict, total=False):
    type: typ.Literal["omat24"]
    """Discriminator for the OMAT24 dataset."""

    src: typ.Required[str]
    """The path to the OMAT24 dataset."""


@typ.overload
def CreateOMAT24DatasetConfig(
    dict: OMAT24DatasetConfigTypedDict, /
) -> OMAT24DatasetConfig: ...


@typ.overload
def CreateOMAT24DatasetConfig(
    **dict: typ.Unpack[OMAT24DatasetConfigTypedDict],
) -> OMAT24DatasetConfig: ...


def CreateOMAT24DatasetConfig(*args, **kwargs):
    from mattertune.data.omat24 import OMAT24DatasetConfig

    dict = args[0] if args else kwargs
    return OMAT24DatasetConfig.model_validate(dict)
