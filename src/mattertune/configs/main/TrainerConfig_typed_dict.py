from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.main import TrainerConfig


__codegen__ = True

# Definitions


class CSVLoggerConfig(typ.TypedDict, total=False):
    type: typ.Literal["csv"]

    save_dir: typ.Required[str]
    """Save directory for logs."""

    name: str
    """Experiment name. Default: ``'lightning_logs'``."""

    version: int | str | None
    """Experiment version. If not specified, automatically assigns the next available version.
    Default: ``None``."""

    prefix: str
    """String to put at the beginning of metric keys. Default: ``''``."""

    flush_logs_every_n_steps: int
    """How often to flush logs to disk. Default: ``100``."""


class EarlyStoppingConfig(typ.TypedDict, total=False):
    monitor: str
    """Quantity to be monitored."""

    min_delta: float
    """Minimum change in monitored quantity to qualify as an improvement. Changes of less than or equal to
    `min_delta` will count as no improvement. Default: ``0.0``."""

    patience: int
    """Number of validation checks with no improvement after which training will be stopped. Default: ``3``."""

    verbose: bool
    """Whether to print messages when improvement is found or early stopping is triggered. Default: ``False``."""

    mode: typ.Literal["min"] | typ.Literal["max"]
    """One of 'min' or 'max'. In 'min' mode, training stops when monitored quantity stops decreasing;
    in 'max' mode it stops when the quantity stops increasing. Default: ``'min'``."""

    strict: bool
    """Whether to raise an error if monitored metric is not found in validation metrics. Default: ``True``."""

    check_finite: bool
    """Whether to stop training when the monitor becomes NaN or infinite. Default: ``True``."""

    stopping_threshold: float | None
    """Stop training immediately once the monitored quantity reaches this threshold. Default: ``None``."""

    divergence_threshold: float | None
    """Stop training as soon as the monitored quantity becomes worse than this threshold. Default: ``None``."""

    check_on_train_epoch_end: bool | None
    """Whether to run early stopping at the end of training epoch. If False, check runs at validation end.
    Default: ``None``."""

    log_rank_zero_only: bool
    """Whether to log the status of early stopping only for rank 0 process. Default: ``False``."""


LoggerConfig = typ.TypeAliasType(
    "LoggerConfig", "CSVLoggerConfig | WandbLoggerConfig | TensorBoardLoggerConfig"
)


class ModelCheckpointConfig(typ.TypedDict, total=False):
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


class TensorBoardLoggerConfigAdditionalParams(typ.TypedDict, total=False):
    """Additional parameters passed to tensorboardX.SummaryWriter. Default: ``{}``."""

    pass


class TensorBoardLoggerConfig(typ.TypedDict, total=False):
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

    additional_params: TensorBoardLoggerConfigAdditionalParams
    """Additional parameters passed to tensorboardX.SummaryWriter. Default: ``{}``."""


class WandbLoggerConfigAdditionalInitParameters(typ.TypedDict, total=False):
    """Additional parameters to pass to wandb.init(). Default: ``{}``."""

    pass


class WandbLoggerConfig(typ.TypedDict, total=False):
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

    additional_init_parameters: WandbLoggerConfigAdditionalInitParameters
    """Additional parameters to pass to wandb.init(). Default: ``{}``."""


# Schema entries
class TrainerConfigTypedDictAdditionalTrainerKwargs(typ.TypedDict, total=False):
    """Additional keyword arguments for the Lightning Trainer.
    This is for advanced users who want to customize the Lightning Trainer,
        and is not recommended for beginners."""

    pass


class TrainerConfigTypedDict(typ.TypedDict, total=False):
    accelerator: str
    """Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
    as well as custom accelerator instances."""

    strategy: str
    """Supports different training strategies with aliases as well custom strategies.
    Default: ``"auto"``."""

    num_nodes: int
    """Number of GPU nodes for distributed training.
    Default: ``1``."""

    devices: list[int] | str | int
    """The devices to use. Can be set to a sequence of device indices, "all" to indicate all available devices should be used, or ``"auto"`` for
    automatic selection based on the chosen accelerator. Default: ``"auto"``."""

    precision: (
        typ.Literal[64]
        | typ.Literal[32]
        | typ.Literal[16]
        | typ.Literal["transformer-engine"]
        | typ.Literal["transformer-engine-float16"]
        | typ.Literal["16-true"]
        | typ.Literal["16-mixed"]
        | typ.Literal["bf16-true"]
        | typ.Literal["bf16-mixed"]
        | typ.Literal["32-true"]
        | typ.Literal["64-true"]
        | typ.Literal["64"]
        | typ.Literal["32"]
        | typ.Literal["16"]
        | typ.Literal["bf16"]
        | None
    )
    """Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
    16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
    Can be used on CPU, GPU, TPUs, HPUs or IPUs.
    Default: ``'32-true'``."""

    deterministic: bool | typ.Literal["warn"] | None
    """If ``True``, sets whether PyTorch operations must use deterministic algorithms.
        Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
        that don't support deterministic mode. If not set, defaults to ``False``. Default: ``None``."""

    max_epochs: int | None
    """Stop training once this number of epochs is reached. Disabled by default (None).
    If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
    To enable infinite training, set ``max_epochs = -1``."""

    min_epochs: int | None
    """Force training for at least these many epochs. Disabled by default (None)."""

    max_steps: int
    """Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
    and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
    ``max_epochs`` to ``-1``."""

    min_steps: int | None
    """Force training for at least these number of steps. Disabled by default (``None``)."""

    max_time: str | str | dict[str, int] | None
    """Stop training after this amount of time has passed. Disabled by default (``None``).
    The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
    :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
    :class:`datetime.timedelta`."""

    val_check_interval: int | float | None
    """How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
    after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
    batches. An ``int`` value can only be higher than the number of training batches when
    ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
    across epochs or during iteration-based training.
    Default: ``1.0``."""

    check_val_every_n_epoch: int | None
    """Perform a validation loop every after every `N` training epochs. If ``None``,
    validation will be done solely based on the number of training batches, requiring ``val_check_interval``
    to be an integer value.
    Default: ``1``."""

    log_every_n_steps: int | None
    """How often to log within steps.
    Default: ``50``."""

    gradient_clip_val: int | float | None
    """The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
    gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
    Default: ``None``."""

    gradient_clip_algorithm: str | None
    """The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
    to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
    be set to ``"norm"``."""

    checkpoint: ModelCheckpointConfig | None
    """The configuration for the model checkpoint."""

    early_stopping: EarlyStoppingConfig | None
    """The configuration for early stopping."""

    loggers: list[LoggerConfig] | typ.Literal["default"]
    """The loggers to use for logging training metrics.
    
    If ``"default"``, will use the CSV logger + the W&B logger if available.
    Default: ``"default"``."""

    additional_trainer_kwargs: TrainerConfigTypedDictAdditionalTrainerKwargs
    """Additional keyword arguments for the Lightning Trainer.
    This is for advanced users who want to customize the Lightning Trainer,
        and is not recommended for beginners."""


@typ.overload
def CreateTrainerConfig(dict: TrainerConfigTypedDict, /) -> TrainerConfig: ...


@typ.overload
def CreateTrainerConfig(
    **dict: typ.Unpack[TrainerConfigTypedDict],
) -> TrainerConfig: ...


def CreateTrainerConfig(*args, **kwargs):
    from mattertune.main import TrainerConfig

    dict = args[0] if args else kwargs
    return TrainerConfig.model_validate(dict)
