from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.loggers import TensorBoardLoggerConfig


__codegen__ = True


# Schema entries
class TensorBoardLoggerConfigTypedDictAdditionalParams(typ.TypedDict, total=False):
    """Additional parameters passed to tensorboardX.SummaryWriter. Default: ``{}``."""

    pass


class TensorBoardLoggerConfigTypedDict(typ.TypedDict, total=False):
    type: typ.Literal["tensorboard"]

    save_dir: typ.Required[str]
    """Save directory where TensorBoard logs will be saved."""

    name: str | None
    """Experiment name. Default: ``'lightning_logs'``. If empty string, no per-experiment subdirectory is used."""

    version: int | str | None
    """Experiment version. If not specified, logger auto-assigns next available version.
    If string, used as run-specific subdirectory name. Default: ``None``."""

    log_graph: bool
    """Whether to add computational graph to tensorboard. Requires model.example_input_array to be defined.
    Default: ``False``."""

    default_hp_metric: bool
    """Enables placeholder metric with key `hp_metric` when logging hyperparameters without a metric.
    Default: ``True``."""

    prefix: str
    """String to put at beginning of metric keys. Default: ``''``."""

    sub_dir: str | None
    """Sub-directory to group TensorBoard logs. If provided, logs are saved in
    ``/save_dir/name/version/sub_dir/``. Default: ``None``."""

    additional_params: TensorBoardLoggerConfigTypedDictAdditionalParams
    """Additional parameters passed to tensorboardX.SummaryWriter. Default: ``{}``."""


@typ.overload
def CreateTensorBoardLoggerConfig(
    dict: TensorBoardLoggerConfigTypedDict, /
) -> TensorBoardLoggerConfig: ...


@typ.overload
def CreateTensorBoardLoggerConfig(
    **dict: typ.Unpack[TensorBoardLoggerConfigTypedDict],
) -> TensorBoardLoggerConfig: ...


def CreateTensorBoardLoggerConfig(*args, **kwargs):
    from mattertune.loggers import TensorBoardLoggerConfig

    dict = args[0] if args else kwargs
    return TensorBoardLoggerConfig.model_validate(dict)
