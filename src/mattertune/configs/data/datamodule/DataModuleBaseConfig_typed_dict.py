from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.datamodule import DataModuleBaseConfig


__codegen__ = True


# Schema entries
class DataModuleBaseConfigTypedDict(typ.TypedDict, total=False):
    batch_size: typ.Required[int]
    """The batch size for the dataloaders."""

    num_workers: int | typ.Literal["auto"]
    """The number of workers for the dataloaders.
    
    This is the number of processes that generate batches in parallel.
    If set to "auto", the number of workers will be automatically
        set based on the number of available CPUs.
    Set to 0 to disable parallelism."""

    pin_memory: bool
    """Whether to pin memory in the dataloaders.
    
    This is useful for speeding up GPU data transfer."""


@typ.overload
def CreateDataModuleBaseConfig(
    dict: DataModuleBaseConfigTypedDict, /
) -> DataModuleBaseConfig: ...


@typ.overload
def CreateDataModuleBaseConfig(
    **dict: typ.Unpack[DataModuleBaseConfigTypedDict],
) -> DataModuleBaseConfig: ...


def CreateDataModuleBaseConfig(*args, **kwargs):
    from mattertune.data.datamodule import DataModuleBaseConfig

    dict = args[0] if args else kwargs
    return DataModuleBaseConfig.model_validate(dict)
