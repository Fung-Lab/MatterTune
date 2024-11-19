from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.datamodule import AutoSplitDataModuleConfig


__codegen__ = True

# Definitions


class DBDatasetConfig(typ.TypedDict, total=False):
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


DatasetConfig = typ.TypeAliasType(
    "DatasetConfig",
    "OMAT24DatasetConfig | XYZDatasetConfig | MPTrajDatasetConfig | MatbenchDatasetConfig | DBDatasetConfig | MPDatasetConfig",
)


class MPDatasetConfigQuery(typ.TypedDict, total=False):
    """Query to filter the data from the Materials Project database."""

    pass


class MPDatasetConfig(typ.TypedDict):
    """Configuration for a dataset stored in the Materials Project database."""

    type: typ.NotRequired[typ.Literal["mp"]]
    """Discriminator for the MP dataset."""

    api: str
    """Input API key for the Materials Project database."""

    fields: list[str]
    """Fields to retrieve from the Materials Project database."""

    query: MPDatasetConfigQuery
    """Query to filter the data from the Materials Project database."""


class MPTrajDatasetConfig(typ.TypedDict, total=False):
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


class MatbenchDatasetConfig(typ.TypedDict, total=False):
    """Configuration for the Matbench dataset."""

    type: typ.Literal["matbench"]
    """Discriminator for the Matbench dataset."""

    task: str | None
    """The name of the self.tasks to include in the dataset."""

    property_name: str | None
    """Assign a property name for the self.task. Must match the property head in the model."""

    fold_idx: (
        typ.Literal[0]
        | typ.Literal[1]
        | typ.Literal[2]
        | typ.Literal[3]
        | typ.Literal[4]
    )
    """The index of the fold to be used in the dataset."""


class OMAT24DatasetConfig(typ.TypedDict, total=False):
    type: typ.Literal["omat24"]
    """Discriminator for the OMAT24 dataset."""

    src: typ.Required[str]
    """The path to the OMAT24 dataset."""


class XYZDatasetConfig(typ.TypedDict, total=False):
    type: typ.Literal["xyz"]
    """Discriminator for the XYZ dataset."""

    src: typ.Required[str | str]
    """The path to the XYZ dataset."""


# Schema entries
class AutoSplitDataModuleConfigTypedDict(typ.TypedDict, total=False):
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

    dataset: typ.Required[DatasetConfig]
    """The configuration for the dataset."""

    train_split: typ.Required[float]
    """The proportion of the dataset to include in the training split."""

    validation_split: float | typ.Literal["auto"] | typ.Literal["disable"]
    """The proportion of the dataset to include in the validation split.
    
    If set to "auto", the validation split will be automatically determined as
    the complement of the training split, i.e. `validation_split = 1 - train_split`.
    
    If set to "disable", the validation split will be disabled."""

    shuffle: bool
    """Whether to shuffle the dataset before splitting."""

    shuffle_seed: int
    """The seed to use for shuffling the dataset."""


@typ.overload
def CreateAutoSplitDataModuleConfig(
    dict: AutoSplitDataModuleConfigTypedDict, /
) -> AutoSplitDataModuleConfig: ...


@typ.overload
def CreateAutoSplitDataModuleConfig(
    **dict: typ.Unpack[AutoSplitDataModuleConfigTypedDict],
) -> AutoSplitDataModuleConfig: ...


def CreateAutoSplitDataModuleConfig(*args, **kwargs):
    from mattertune.data.datamodule import AutoSplitDataModuleConfig

    dict = args[0] if args else kwargs
    return AutoSplitDataModuleConfig.model_validate(dict)
