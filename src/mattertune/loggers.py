from __future__ import annotations

from typing import Annotated, Any, Literal

import nshconfig as C
from typing_extensions import TypeAliasType


class CSVLoggerConfig(C.Config):
    type: Literal["csv"] = "csv"

    save_dir: str
    """Save directory for logs."""

    name: str = "lightning_logs"
    """Experiment name. Default: ``'lightning_logs'``."""

    version: int | str | None = None
    """Experiment version. If not specified, automatically assigns the next available version.
    Default: ``None``."""

    prefix: str = ""
    """String to put at the beginning of metric keys. Default: ``''``."""

    flush_logs_every_n_steps: int = 100
    """How often to flush logs to disk. Default: ``100``."""

    def create_logger(self):
        """Creates a CSVLogger instance from this config."""
        from lightning.pytorch.loggers.csv_logs import CSVLogger

        return CSVLogger(
            save_dir=self.save_dir,
            name=self.name,
            version=self.version,
            prefix=self.prefix,
            flush_logs_every_n_steps=self.flush_logs_every_n_steps,
        )


class WandbLoggerConfig(C.Config):
    type: Literal["wandb"] = "wandb"

    name: str | None = None
    """Display name for the run. Default: ``None``."""

    save_dir: str = "."
    """Path where data is saved. Default: ``.``."""

    version: str | None = None
    """Sets the version, mainly used to resume a previous run. Default: ``None``."""

    offline: bool = False
    """Run offline (data can be streamed later to wandb servers). Default: ``False``."""

    dir: str | None = None
    """Same as save_dir. Default: ``None``."""

    id: str | None = None
    """Same as version. Default: ``None``."""

    anonymous: bool | None = None
    """Enables or explicitly disables anonymous logging. Default: ``None``."""

    project: str | None = None
    """The name of the project to which this run will belong. Default: ``None``."""

    log_model: Literal["all"] | bool = False
    """Whether/how to log model checkpoints as W&B artifacts. Default: ``False``.
    If 'all', checkpoints are logged during training.
    If True, checkpoints are logged at the end of training.
    If False, no checkpoints are logged."""

    prefix: str = ""
    """A string to put at the beginning of metric keys. Default: ``''``."""

    experiment: Any | None = None  # Run | RunDisabled | None
    """WandB experiment object. Automatically set when creating a run. Default: ``None``."""

    checkpoint_name: str | None = None
    """Name of the model checkpoint artifact being logged. Default: ``None``."""

    additional_init_parameters: dict[str, Any] = {}
    """Additional parameters to pass to wandb.init(). Default: ``{}``."""

    def create_logger(self):
        """Creates a WandbLogger instance from this config."""
        from lightning.pytorch.loggers.wandb import WandbLogger

        # Pass all parameters except additional_init_parameters to constructor
        base_params = {
            k: v
            for k, v in self.model_dump().items()
            if k != "additional_init_parameters" and k != "type"
        }

        # Merge with additional init parameters
        return WandbLogger(**base_params, **self.additional_init_parameters)


class TensorBoardLoggerConfig(C.Config):
    type: Literal["tensorboard"] = "tensorboard"

    save_dir: str
    """Save directory where TensorBoard logs will be saved."""

    name: str | None = "lightning_logs"
    """Experiment name. Default: ``'lightning_logs'``. If empty string, no per-experiment subdirectory is used."""

    version: int | str | None = None
    """Experiment version. If not specified, logger auto-assigns next available version.
    If string, used as run-specific subdirectory name. Default: ``None``."""

    log_graph: bool = False
    """Whether to add computational graph to tensorboard. Requires model.example_input_array to be defined.
    Default: ``False``."""

    default_hp_metric: bool = True
    """Enables placeholder metric with key `hp_metric` when logging hyperparameters without a metric.
    Default: ``True``."""

    prefix: str = ""
    """String to put at beginning of metric keys. Default: ``''``."""

    sub_dir: str | None = None
    """Sub-directory to group TensorBoard logs. If provided, logs are saved in
    ``/save_dir/name/version/sub_dir/``. Default: ``None``."""

    additional_params: dict[str, Any] = {}
    """Additional parameters passed to tensorboardX.SummaryWriter. Default: ``{}``."""

    def create_logger(self):
        """Creates a TensorBoardLogger instance from this config."""
        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

        # Pass all parameters except additional_params to constructor
        base_params = {
            k: v
            for k, v in self.model_dump().items()
            if k != "additional_params" and k != "type"
        }

        # Merge with additional tensorboard parameters
        return TensorBoardLogger(**base_params, **self.additional_params)


LoggerConfig = TypeAliasType(
    "LoggerConfig",
    Annotated[
        CSVLoggerConfig | WandbLoggerConfig | TensorBoardLoggerConfig,
        C.Field(description="Logger configuration.", discriminator="type"),
    ],
)
