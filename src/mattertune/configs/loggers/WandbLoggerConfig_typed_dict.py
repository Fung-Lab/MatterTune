from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.loggers import WandbLoggerConfig


__codegen__ = True


# Schema entries
class WandbLoggerConfigTypedDictAdditionalInitParameters(typ.TypedDict, total=False):
    """Additional parameters to pass to wandb.init(). Default: ``{}``."""

    pass


class WandbLoggerConfigTypedDict(typ.TypedDict, total=False):
    type: typ.Literal["wandb"]

    name: str | None
    """Display name for the run. Default: ``None``."""

    save_dir: str
    """Path where data is saved. Default: ``.``."""

    version: str | None
    """Sets the version, mainly used to resume a previous run. Default: ``None``."""

    offline: bool
    """Run offline (data can be streamed later to wandb servers). Default: ``False``."""

    dir: str | None
    """Same as save_dir. Default: ``None``."""

    id: str | None
    """Same as version. Default: ``None``."""

    anonymous: bool | None
    """Enables or explicitly disables anonymous logging. Default: ``None``."""

    project: str | None
    """The name of the project to which this run will belong. Default: ``None``."""

    log_model: typ.Literal["all"] | bool
    """Whether/how to log model checkpoints as W&B artifacts. Default: ``False``.
    If 'all', checkpoints are logged during training.
    If True, checkpoints are logged at the end of training.
    If False, no checkpoints are logged."""

    prefix: str
    """A string to put at the beginning of metric keys. Default: ``''``."""

    experiment: typ.Any | None
    """WandB experiment object. Automatically set when creating a run. Default: ``None``."""

    checkpoint_name: str | None
    """Name of the model checkpoint artifact being logged. Default: ``None``."""

    additional_init_parameters: WandbLoggerConfigTypedDictAdditionalInitParameters
    """Additional parameters to pass to wandb.init(). Default: ``{}``."""


@typ.overload
def CreateWandbLoggerConfig(
    dict: WandbLoggerConfigTypedDict, /
) -> WandbLoggerConfig: ...


@typ.overload
def CreateWandbLoggerConfig(
    **dict: typ.Unpack[WandbLoggerConfigTypedDict],
) -> WandbLoggerConfig: ...


def CreateWandbLoggerConfig(*args, **kwargs):
    from mattertune.loggers import WandbLoggerConfig

    dict = args[0] if args else kwargs
    return WandbLoggerConfig.model_validate(dict)
