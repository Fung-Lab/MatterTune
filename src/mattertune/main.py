from __future__ import annotations

import logging
from typing import Any

import nshconfig as C
from lightning.pytorch import Trainer

from .backbones import ModelConfig
from .data import DataModuleConfig, MatterTuneDataModule
from .finetune.base import FinetuneModuleBase
from .registry import backbone_registry, data_registry

log = logging.getLogger(__name__)


@backbone_registry.rebuild_on_registers
@data_registry.rebuild_on_registers
class MatterTunerConfig(C.Config):
    data: DataModuleConfig
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

        # Create the datamodule
        datamodule = MatterTuneDataModule(self.config.data)

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
        trainer.fit(lightning_module, datamodule)

        # Return the trained model
        return lightning_module
