from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.matbench import MatbenchDatasetConfig


__codegen__ = True

"""Configuration for the Matbench dataset."""


# Schema entries
class MatbenchDatasetConfigTypedDict(typ.TypedDict, total=False):
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


@typ.overload
def CreateMatbenchDatasetConfig(
    dict: MatbenchDatasetConfigTypedDict, /
) -> MatbenchDatasetConfig: ...


@typ.overload
def CreateMatbenchDatasetConfig(
    **dict: typ.Unpack[MatbenchDatasetConfigTypedDict],
) -> MatbenchDatasetConfig: ...


def CreateMatbenchDatasetConfig(*args, **kwargs):
    from mattertune.data.matbench import MatbenchDatasetConfig

    dict = args[0] if args else kwargs
    return MatbenchDatasetConfig.model_validate(dict)
