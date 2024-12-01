from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, Literal, NamedTuple

import nshconfig as C
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer
from lightning.pytorch.strategies.strategy import Strategy

from .backbones import ModelConfig
from .callbacks.early_stopping import EarlyStoppingConfig
from .callbacks.model_checkpoint import ModelCheckpointConfig
from .data import DataModuleConfig, MatterTuneDataModule
from .finetune.base import FinetuneModuleBase
from .loggers import CSVLoggerConfig, LoggerConfig
from .registry import backbone_registry, data_registry

log = logging.getLogger(__name__)


class TuneOutput(NamedTuple):
    """The output of the MatterTuner.tune method."""

    model: FinetuneModuleBase
    """The trained model."""

    trainer: Trainer
    """The trainer used to train the model."""


class TrainerConfig(C.Config):
    accelerator: str = "auto"
    """Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
    as well as custom accelerator instances."""

    strategy: str | Strategy = "auto"
    """Supports different training strategies with aliases as well custom strategies.
    Default: ``"auto"``.
    """

    num_nodes: int = 1
    """Number of GPU nodes for distributed training.
    Default: ``1``.
    """

    devices: list[int] | str | int = "auto"
    """The devices to use. Can be set to a sequence of device indices, "all" to indicate all available devices should be used, or ``"auto"`` for
    automatic selection based on the chosen accelerator. Default: ``"auto"``.
    """

    precision: _PRECISION_INPUT | None = "32-true"
    """Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
    16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
    Can be used on CPU, GPU, TPUs, HPUs or IPUs.
    Default: ``'32-true'``."""

    deterministic: bool | Literal["warn"] | None = None
    """
    If ``True``, sets whether PyTorch operations must use deterministic algorithms.
        Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
        that don't support deterministic mode. If not set, defaults to ``False``. Default: ``None``.
    """

    max_epochs: int | None = None
    """Stop training once this number of epochs is reached. Disabled by default (None).
    If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
    To enable infinite training, set ``max_epochs = -1``."""

    min_epochs: int | None = None
    """Force training for at least these many epochs. Disabled by default (None)."""

    max_steps: int = -1
    """Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
    and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
    ``max_epochs`` to ``-1``."""

    min_steps: int | None = None
    """Force training for at least these number of steps. Disabled by default (``None``)."""

    max_time: str | timedelta | dict[str, int] | None = None
    """Stop training after this amount of time has passed. Disabled by default (``None``).
    The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
    :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
    :class:`datetime.timedelta`."""

    val_check_interval: int | float | None = None
    """How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
    after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
    batches. An ``int`` value can only be higher than the number of training batches when
    ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
    across epochs or during iteration-based training.
    Default: ``1.0``.
    """

    check_val_every_n_epoch: int | None = 1
    """Perform a validation loop every after every `N` training epochs. If ``None``,
    validation will be done solely based on the number of training batches, requiring ``val_check_interval``
    to be an integer value.
    Default: ``1``.
    """

    log_every_n_steps: int | None = None
    """How often to log within steps.
    Default: ``50``.
    """

    gradient_clip_val: int | float | None = None
    """The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
    gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
    Default: ``None``.
    """

    gradient_clip_algorithm: str | None = None
    """The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
    to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
    be set to ``"norm"``.
    """

    checkpoint: ModelCheckpointConfig | None = None
    """The configuration for the model checkpoint."""

    early_stopping: EarlyStoppingConfig | None = None
    """The configuration for early stopping."""

    loggers: Sequence[LoggerConfig] | Literal["default"] = "default"
    """The loggers to use for logging training metrics.

    If ``"default"``, will use the CSV logger + the W&B logger if available.
    Default: ``"default"``.
    """

    additional_trainer_kwargs: dict[str, Any] = {}
    """
    Additional keyword arguments for the Lightning Trainer.

    This is for advanced users who want to customize the Lightning Trainer,
    and is not recommended for beginners.
    """

    def _to_lightning_kwargs(self):
        callbacks = []
        if self.checkpoint is not None:
            callbacks.append(self.checkpoint.create_callback())
        if self.early_stopping is not None:
            callbacks.append(self.early_stopping.create_callback())

        loggers = []
        if self.loggers == "default":
            loggers.append(CSVLoggerConfig(save_dir="./logs").create_logger())
        else:
            for logger_config in self.loggers:
                loggers.append(logger_config.create_logger())

        kwargs = {
            "callbacks": callbacks,
            "accelerator": self.accelerator,
            "strategy": self.strategy,
            "devices": self.devices,
            "num_nodes": self.num_nodes,
            "precision": self.precision,
            "deterministic": self.deterministic,
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "max_steps": self.max_steps,
            "min_steps": self.min_steps,
            "max_time": self.max_time,
            "val_check_interval": self.val_check_interval,
            "check_val_every_n_epoch": self.check_val_every_n_epoch,
            "log_every_n_steps": self.log_every_n_steps,
            "gradient_clip_val": self.gradient_clip_val,
            "gradient_clip_algorithm": self.gradient_clip_algorithm,
            "logger": loggers,
        }

        # Add the additional trainer kwargs
        kwargs.update(self.additional_trainer_kwargs)

        return kwargs


@backbone_registry.rebuild_on_registers
@data_registry.rebuild_on_registers
class MatterTunerConfig(C.Config):
    data: DataModuleConfig
    """The configuration for the data."""

    model: ModelConfig
    """The configuration for the model."""

    trainer: TrainerConfig = TrainerConfig()
    """The configuration for the trainer."""


class MatterTuner:
    def __init__(self, config: MatterTunerConfig):
        self.config = config

    def tune(self, trainer_kwargs: dict[str, Any] | None = None) -> TuneOutput:
        # Make sure all the necessary dependencies are installed
        self.config.model.ensure_dependencies()

        # Create the model
        lightning_module = self.config.model.create_model()
        assert isinstance(
            lightning_module, FinetuneModuleBase
        ), f'The backbone model must be a FinetuneModuleBase subclass. Got "{type(lightning_module)}".'

        # Create the datamodule
        datamodule = MatterTuneDataModule(self.config.data)

        # Resolve the full trainer kwargs
        trainer_kwargs_: dict[str, Any] = self.config.trainer._to_lightning_kwargs()

        # Update with the user-specified kwargs in the method call
        if trainer_kwargs is not None:
            trainer_kwargs_.update(trainer_kwargs)

        if lightning_module.requires_disabled_inference_mode():
            if (
                user_inference_mode := trainer_kwargs_.get("inference_mode")
            ) is not None and user_inference_mode:
                raise ValueError(
                    "The model requires inference_mode to be disabled. "
                    "But the provided trainer kwargs have inference_mode=True. "
                    "Please set inference_mode=False.\n"
                    "If you think this is a mistake, please report a bug."
                )

            log.info(
                "The model requires inference_mode to be disabled. "
                "Setting inference_mode=False."
            )
            trainer_kwargs_["inference_mode"] = False

        # Create the trainer
        trainer = Trainer(**trainer_kwargs_)
        trainer.fit(lightning_module, datamodule)

        # Return the trained model
        return TuneOutput(model=lightning_module, trainer=trainer)
