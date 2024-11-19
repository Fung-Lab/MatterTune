from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.callbacks.model_checkpoint import ModelCheckpointConfig


__codegen__ = True


# Schema entries
class ModelCheckpointConfigTypedDict(typ.TypedDict, total=False):
    dirpath: str | None
    """Directory to save the model file. Default: ``None``."""

    filename: str | None
    """Checkpoint filename. Can contain named formatting options. Default: ``None``."""

    monitor: str | None
    """Quantity to monitor. Default: ``None``."""

    verbose: bool
    """Verbosity mode. Default: ``False``."""

    save_last: typ.Literal[True] | typ.Literal[False] | typ.Literal["link"] | None
    """When True or "link", saves a 'last.ckpt' checkpoint when a checkpoint is saved. Default: ``None``."""

    save_top_k: int
    """If save_top_k=k, save k models with best monitored quantity. Default: ``1``."""

    save_weights_only: bool
    """If True, only save model weights. Default: ``False``."""

    mode: typ.Literal["min"] | typ.Literal["max"]
    """One of {'min', 'max'}. For 'min' training stops when monitored quantity stops decreasing. Default: ``'min'``."""

    auto_insert_metric_name: bool
    """Whether to automatically insert metric name in checkpoint filename. Default: ``True``."""

    every_n_train_steps: int | None
    """Number of training steps between checkpoints. Default: ``None``."""

    train_time_interval: str | None
    """Checkpoints are monitored at the specified time interval. Default: ``None``."""

    every_n_epochs: int | None
    """Number of epochs between checkpoints. Default: ``None``."""

    save_on_train_epoch_end: bool | None
    """Whether to run checkpointing at end of training epoch. Default: ``None``."""

    enable_version_counter: bool
    """Whether to append version to existing filenames. Default: ``True``."""


@typ.overload
def CreateModelCheckpointConfig(
    dict: ModelCheckpointConfigTypedDict, /
) -> ModelCheckpointConfig: ...


@typ.overload
def CreateModelCheckpointConfig(
    **dict: typ.Unpack[ModelCheckpointConfigTypedDict],
) -> ModelCheckpointConfig: ...


def CreateModelCheckpointConfig(*args, **kwargs):
    from mattertune.callbacks.model_checkpoint import ModelCheckpointConfig

    dict = args[0] if args else kwargs
    return ModelCheckpointConfig.model_validate(dict)
