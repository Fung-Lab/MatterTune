from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.base import DatasetConfigBase


__codegen__ = True


# Schema entries
class DatasetConfigBaseTypedDict(typ.TypedDict, total=False):
    pass


@typ.overload
def CreateDatasetConfigBase(
    dict: DatasetConfigBaseTypedDict, /
) -> DatasetConfigBase: ...


@typ.overload
def CreateDatasetConfigBase(
    **dict: typ.Unpack[DatasetConfigBaseTypedDict],
) -> DatasetConfigBase: ...


def CreateDatasetConfigBase(*args, **kwargs):
    from mattertune.data.base import DatasetConfigBase

    dict = args[0] if args else kwargs
    return DatasetConfigBase.model_validate(dict)
