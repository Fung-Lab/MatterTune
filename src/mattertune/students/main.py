from __future__ import annotations

import logging
from typing import Any

import nshconfig as C
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback

from . import StudentModelConfig
from ..data import DataModuleConfig, MatterTuneDataModule
from ..distillation.base import StudentModuleBase
from ..registry import student_registry, data_registry
from ..main import TrainerConfig


log = logging.getLogger(__name__)


@student_registry.rebuild_on_registers
@data_registry.rebuild_on_registers
class OfflineDistillationTrainerConfig(C.Config):
    """
    Offline distillation framework for training student models using a pre-trained foundation model.
    A general workflow is as follows:
    1. Generate and label synthetic data using the foundation model.
    2. Train the student model using the labeled synthetic data.
    
    To avoid env conficts between foundation model and student model,
    the data generation part and the student training part are separated.
    
    Here, we implement the student training part.
    """
    data: DataModuleConfig
    """The configuration for the data."""

    model: StudentModelConfig
    """The configuration for the model."""

    trainer: TrainerConfig = TrainerConfig()
    """The configuration for the trainer."""
    

class MatterTuneOfflineDistillationTrainer:
    """
    Offline distillation framework for training student models using a pre-trained foundation model.
    A general workflow is as follows:
    1. Generate and label synthetic data using the foundation model.
    2. Train the student model using the labeled synthetic data.
    
    To avoid env conficts between foundation model and student model,
    the data generation part and the student training part are separated.
    
    Here, we implement the student training part.
    """
    def __init__(
        self,
        config: OfflineDistillationTrainerConfig,
    ):
        self.config = config
        
    def train(
        self,
        trainer_kwargs: dict[str, Any] | None = None
    ) -> StudentModuleBase:
        # Make sure all the necessary dependencies are installed
        self.config.model.ensure_dependencies()

        # Create the model
        lightning_module = self.config.model.create_model()
        assert isinstance(lightning_module, StudentModuleBase), f"Model must be an instance of StudentModuleBase, but got {type(lightning_module)}"
        
        # Create the datamodule
        datamodule = MatterTuneDataModule(self.config.data)

        # Resolve the full trainer kwargs
        trainer_kwargs_: dict[str, Any] = self.config.trainer._to_lightning_kwargs()
        
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
            
        # Set up the callbacks for recipes
        callbacks: list[Callback] = trainer_kwargs_.pop("callbacks", [])
        trainer_kwargs_["callbacks"] = callbacks
        
        # set up the data and model
        try:
            datamodule.prepare_data()
        except Exception:
            pass
        datamodule.setup(stage="fit")
        
        lightning_module.before_fit_start(datamodule)

        # Create the trainer
        trainer = Trainer(**trainer_kwargs_)
        trainer.fit(lightning_module, datamodule)
        
        return lightning_module
