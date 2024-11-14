from __future__ import annotations

from typing import Literal

import nshconfig as C


class EarlyStoppingConfig(C.Config):
    monitor: str = "val/total_loss"
    """Quantity to be monitored."""

    min_delta: float = 0.0
    """Minimum change in monitored quantity to qualify as an improvement. Changes of less than or equal to
    `min_delta` will count as no improvement. Default: ``0.0``."""

    patience: int = 3
    """Number of validation checks with no improvement after which training will be stopped. Default: ``3``."""

    verbose: bool = False
    """Whether to print messages when improvement is found or early stopping is triggered. Default: ``False``."""

    mode: Literal["min", "max"] = "min"
    """One of 'min' or 'max'. In 'min' mode, training stops when monitored quantity stops decreasing;
    in 'max' mode it stops when the quantity stops increasing. Default: ``'min'``."""

    strict: bool = True
    """Whether to raise an error if monitored metric is not found in validation metrics. Default: ``True``."""

    check_finite: bool = True
    """Whether to stop training when the monitor becomes NaN or infinite. Default: ``True``."""

    stopping_threshold: float | None = None
    """Stop training immediately once the monitored quantity reaches this threshold. Default: ``None``."""

    divergence_threshold: float | None = None
    """Stop training as soon as the monitored quantity becomes worse than this threshold. Default: ``None``."""

    check_on_train_epoch_end: bool | None = None
    """Whether to run early stopping at the end of training epoch. If False, check runs at validation end.
    Default: ``None``."""

    log_rank_zero_only: bool = False
    """Whether to log the status of early stopping only for rank 0 process. Default: ``False``."""

    def create_callback(self):
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping

        """Creates an EarlyStopping callback instance from this config."""
        return EarlyStopping(
            monitor=self.monitor,
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=self.verbose,
            mode=self.mode,
            strict=self.strict,
            check_finite=self.check_finite,
            stopping_threshold=self.stopping_threshold,
            divergence_threshold=self.divergence_threshold,
            check_on_train_epoch_end=self.check_on_train_epoch_end,
            log_rank_zero_only=self.log_rank_zero_only,
        )
