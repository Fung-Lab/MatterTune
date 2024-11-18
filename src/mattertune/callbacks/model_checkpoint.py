from __future__ import annotations

from datetime import timedelta
from typing import Literal

import nshconfig as C


class ModelCheckpointConfig(C.Config):
    dirpath: str | None = None
    """Directory to save the model file. Default: ``None``."""

    filename: str | None = None
    """Checkpoint filename. Can contain named formatting options. Default: ``None``."""

    monitor: str | None = None
    """Quantity to monitor. Default: ``None``."""

    verbose: bool = False
    """Verbosity mode. Default: ``False``."""

    save_last: Literal[True, False, "link"] | None = None
    """When True or "link", saves a 'last.ckpt' checkpoint when a checkpoint is saved. Default: ``None``."""

    save_top_k: int = 1
    """If save_top_k=k, save k models with best monitored quantity. Default: ``1``."""

    save_weights_only: bool = False
    """If True, only save model weights. Default: ``False``."""

    mode: Literal["min", "max"] = "min"
    """One of {'min', 'max'}. For 'min' training stops when monitored quantity stops decreasing. Default: ``'min'``."""

    auto_insert_metric_name: bool = True
    """Whether to automatically insert metric name in checkpoint filename. Default: ``True``."""

    every_n_train_steps: int | None = None
    """Number of training steps between checkpoints. Default: ``None``."""

    train_time_interval: timedelta | None = None
    """Checkpoints are monitored at the specified time interval. Default: ``None``."""

    every_n_epochs: int | None = None
    """Number of epochs between checkpoints. Default: ``None``."""

    save_on_train_epoch_end: bool | None = None
    """Whether to run checkpointing at end of training epoch. Default: ``None``."""

    enable_version_counter: bool = True
    """Whether to append version to existing filenames. Default: ``True``."""

    def create_callback(self):
        """Creates a ModelCheckpoint callback instance from this config."""
        from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

        return ModelCheckpoint(
            dirpath=self.dirpath,
            filename=self.filename,
            monitor=self.monitor,
            verbose=self.verbose,
            save_last=self.save_last,
            save_top_k=self.save_top_k,
            save_weights_only=self.save_weights_only,
            mode=self.mode,
            auto_insert_metric_name=self.auto_insert_metric_name,
            every_n_train_steps=self.every_n_train_steps,
            train_time_interval=self.train_time_interval,
            every_n_epochs=self.every_n_epochs,
            save_on_train_epoch_end=self.save_on_train_epoch_end,
            enable_version_counter=self.enable_version_counter,
        )
