from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Literal

import nshconfig as C
from lightning.pytorch import Trainer

from ..backbones import ModelConfig
from ..data import DatasetConfig
from ..finetune.base import FinetuneModuleBase
from ..registry import backbone_registry, data_registry

if TYPE_CHECKING:
    from .loader import DataLoaderKwargs

log = logging.getLogger(__name__)


@data_registry.rebuild_on_registers
class PerSplitDataConfig(C.Config):
    train: DatasetConfig
    """The configuration for the training data."""

    validation: DatasetConfig | None = None
    """The configuration for the validation data."""

    batch_size: int
    """The batch size for the dataloaders.
    TODO: Add support for auto batch size tuning."""

    num_workers: int | Literal["auto"] = "auto"
    """The number of workers for the dataloaders.

    This is the number of processes that generate batches in parallel.
    If set to "auto", the number of workers will be automatically
        set based on the number of available CPUs.
    Set to 0 to disable parallelism.
    """

    pin_memory: bool = True
    """Whether to pin memory in the dataloaders.

    This is useful for speeding up GPU data transfer.
    """

    def _num_workers_or_auto(self):
        if self.num_workers == "auto":
            import os

            if (cpu_count := os.cpu_count()) is not None:
                return cpu_count - 1
            else:
                return 1

        return self.num_workers

    def dataloader_kwargs(self) -> DataLoaderKwargs:
        return {
            "batch_size": self.batch_size,
            "num_workers": self._num_workers_or_auto(),
            "pin_memory": self.pin_memory,
        }


@backbone_registry.rebuild_on_registers
@data_registry.rebuild_on_registers
class MatterTunerConfig(C.Config):
    data: PerSplitDataConfig
    """The configuration for the data."""

    model: ModelConfig
    """The configuration for the model."""

    lightning_trainer_kwargs: dict[str, Any] = {}
    """
    Keyword arguments for the Lightning Trainer.
    This is for advanced users who want to customize the Lightning Trainer,
        and is not recommended for beginners.
    """


class MatterTuner:
    def __init__(self, config: MatterTunerConfig):
        self.config = config

    def tune(self):
        # Resolve the model class
        model_cls = self.config.model.model_cls()

        # Make sure all the necessary dependencies are installed
        model_cls.ensure_dependencies()

        # Create the model
        lightning_module = model_cls(self.config.model)
        assert isinstance(
            lightning_module, FinetuneModuleBase
        ), f'The backbone model must be a FinetuneModuleBase subclass. Got "{type(lightning_module)}".'

        # Create the datasets & wrap them in our dataloader logic
        train_dataloader = _create_dataset(
            lightning_module,
            self.config.data,
            self.config.data.train,
            has_labels=True,
        )
        val_dataloader = (
            _create_dataset(
                lightning_module,
                self.config.data,
                self.config.data.validation,
                has_labels=True,
            )
            if self.config.data.validation is not None
            else None
        )

        # Resolve the full trainer kwargs
        trainer_kwargs: dict[str, Any] = {}
        if lightning_module.requires_disabled_inference_mode():
            if (
                user_inference_mode := self.config.lightning_trainer_kwargs.get(
                    "inference_mode"
                )
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
            trainer_kwargs["inference_mode"] = False
        # Update with the user-specified kwargs
        trainer_kwargs.update(self.config.lightning_trainer_kwargs)

        # Create the trainer
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(
            lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Return the trained model
        return lightning_module


def _create_dataset(
    lightning_module: FinetuneModuleBase,
    data_hparams: PerSplitDataConfig,
    dataset_hparams: DatasetConfig,
    has_labels: bool,
):
    # Ensure all dependencies are installed for the dataset
    dataset_hparams.ensure_dependencies()

    # Create the dataset and wrap it in our dataloader logic
    dataset = dataset_hparams.create_dataset()

    return lightning_module.create_dataloader(
        dataset,
        has_labels=has_labels,
        **data_hparams.dataloader_kwargs(),
    )
