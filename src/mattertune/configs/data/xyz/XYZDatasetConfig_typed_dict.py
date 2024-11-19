from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.xyz import XYZDatasetConfig


__codegen__ = True


# Schema entries
class XYZDatasetConfigTypedDict(typ.TypedDict, total=False):
    type: typ.Literal["xyz"]
    """Discriminator for the XYZ dataset."""

    src: typ.Required[str | str]
    """The path to the XYZ dataset."""


@typ.overload
def CreateXYZDatasetConfig(dict: XYZDatasetConfigTypedDict, /) -> XYZDatasetConfig: ...


@typ.overload
def CreateXYZDatasetConfig(
    **dict: typ.Unpack[XYZDatasetConfigTypedDict],
) -> XYZDatasetConfig: ...


def CreateXYZDatasetConfig(*args, **kwargs):
    from mattertune.data.xyz import XYZDatasetConfig

    dict = args[0] if args else kwargs
    return XYZDatasetConfig.model_validate(dict)
