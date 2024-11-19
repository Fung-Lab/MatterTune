from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.callbacks.early_stopping import EarlyStoppingConfig


__codegen__ = True


# Schema entries
class EarlyStoppingConfigTypedDict(typ.TypedDict, total=False):
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


@typ.overload
def CreateEarlyStoppingConfig(
    dict: EarlyStoppingConfigTypedDict, /
) -> EarlyStoppingConfig: ...


@typ.overload
def CreateEarlyStoppingConfig(
    **dict: typ.Unpack[EarlyStoppingConfigTypedDict],
) -> EarlyStoppingConfig: ...


def CreateEarlyStoppingConfig(*args, **kwargs):
    from mattertune.callbacks.early_stopping import EarlyStoppingConfig

    dict = args[0] if args else kwargs
    return EarlyStoppingConfig.model_validate(dict)
