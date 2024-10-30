from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Generic

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict, TypeVar, cast, override

from .loss import (
    HuberLossConfig,
    L2MAELossConfig,
    MAELossConfig,
    MSELossConfig,
    l2_mae_loss,
)
from .lr_scheduler import LRSchedulerConfig
from .metrics import FinetuneMetrics
from .optimizer import OptimizerConfig
from .properties import PropertyConfig

if TYPE_CHECKING:
    from ase import Atoms

log = logging.getLogger(__name__)


class FinetuneModuleBaseConfig(BaseModel):
    properties: Mapping[str, PropertyConfig]
    """Properties to predict."""

    optimizer: OptimizerConfig
    """Optimizer."""

    lr_scheduler: LRSchedulerConfig
    """Learning Rate Scheduler"""

    ignore_gpu_batch_transform_error: bool = True
    """Whether to ignore data processing errors during training."""


TFinetuneModuleConfig = TypeVar("TFinetuneModuleConfig", bound=FinetuneModuleBaseConfig)


class _SkipBatchError(Exception):
    pass


class ModelPrediction(TypedDict):
    predicted_properties: dict[str, torch.Tensor]
    """Predicted properties."""

    backbone_output: NotRequired[Any]
    """Output of the backbone model. Only set if `return_backbone_output` is True."""


TData = TypeVar("TData")
TBatch = TypeVar("TBatch")


class FinetuneModuleBase(
    LightningModule,
    ABC,
    Generic[TData, TBatch, TFinetuneModuleConfig],
):
    """
    Finetune module base class. Inherits ``lightning.pytorch.LightningModule``.
    """

    # region ABC methods for output heads and model forward pass
    @abstractmethod
    def create_model(self):
        """
        Initialize both the pre-trained backbone and the
        output heads for the properties to predict.
        """
        ...

    @abstractmethod
    def model_forward_context(self, data: TBatch) -> contextlib.AbstractContextManager:
        """
        Context manager for the model forward pass.

        This is used for any setup that needs to be done before the forward pass,
            e.g., setting pos.requires_grad_() for gradient-based force prediction.
        """
        ...

    @abstractmethod
    def model_forward(
        self,
        batch: TBatch,
        return_backbone_output: bool = False,
    ) -> ModelPrediction:
        """
        Forward pass of the model.

        Args:
            batch: Input batch.
            return_backbone_output: Whether to return the output of the backbone model.

        Returns:
            Prediction of the model.
        """
        ...

    @abstractmethod
    def pretrained_backbone_parameters(self) -> Iterable[nn.Parameter]:
        """
        Return the parameters of the backbone model.

        Default implementation returns all parameters that are not part of any output head.
        """
        ...

    # endregion

    # region ABC methods for data processing
    @abstractmethod
    def cpu_data_transform(self, data: TData) -> TData:
        """
        Transform data (on the CPU) before being batched and sent to the GPU.
        """
        ...

    @abstractmethod
    def collate_fn(self, data_list: list[TData]) -> TBatch:
        """
        Collate function for the DataLoader
        """
        ...

    @abstractmethod
    def gpu_batch_transform(self, batch: TBatch) -> TBatch:
        """
        Transform batch (on the GPU) before being fed to the model.
        """
        ...

    @abstractmethod
    def batch_to_labels(self, batch: TBatch) -> dict[str, torch.Tensor]:
        """
        Extract ground truth values from a batch. The output of this function
        should be a dictionary with keys corresponding to the target names
        and values corresponding to the ground truth values. The values should
        be torch tensors that match, in shape, the output of the corresponding
        output head.
        """
        ...

    @abstractmethod
    def atoms_to_data(self, atoms: Atoms) -> TData:
        """
        Convert an ASE atoms object to a data object.
        """
        ...

    # endregion

    hparams: TFinetuneModuleConfig  # pyright: ignore[reportIncompatibleMethodOverride]
    hparams_initial: TFinetuneModuleConfig  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, hparams: TFinetuneModuleConfig):
        super().__init__()

        # Save the hyperparameters
        self.save_hyperparameters(hparams)

        # Create the backbone model and output heads
        self.create_model()

        # Create metrics
        self.create_metrics()

    def create_metrics(self):
        self.train_metrics = FinetuneMetrics(self.hparams.properties)
        self.val_metrics = FinetuneMetrics(self.hparams.properties)
        self.test_metrics = FinetuneMetrics(self.hparams.properties)

        if not any(p for p in self.parameters() if p.requires_grad):
            raise ValueError(
                "No parameters require gradients. "
                "Please ensure that some parts of the model are trainable."
            )

    @override
    def forward(
        self,
        batch: TBatch,
        return_backbone_output: bool = False,
        ignore_gpu_batch_transform_error: bool | None = None,
    ) -> ModelPrediction:
        if ignore_gpu_batch_transform_error is None:
            ignore_gpu_batch_transform_error = (
                self.hparams.ignore_gpu_batch_transform_error
            )

        with self.model_forward_context(batch):
            # Generate graph/etc
            if ignore_gpu_batch_transform_error:
                try:
                    batch = self.gpu_batch_transform(batch)
                except Exception as e:
                    log.warning("Error in forward pass. Skipping batch.", exc_info=e)
                    raise _SkipBatchError() from e
            else:
                batch = self.gpu_batch_transform(batch)

            # Run the model
            return self.model_forward(
                batch, return_backbone_output=return_backbone_output
            )

    def _compute_loss_for_head(
        self,
        hparams: PropertyConfig,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ):
        match hparams.loss:
            case MAELossConfig():
                return F.l1_loss(prediction, target)
            case MSELossConfig():
                return F.mse_loss(prediction, target)
            case HuberLossConfig():
                return F.huber_loss(prediction, target, delta=hparams.loss.delta)
            case L2MAELossConfig():
                return l2_mae_loss(prediction, target)
            case _:
                raise ValueError(f"Unknown loss function: {hparams.loss}")

    def _compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        log: bool = True,
        log_prefix: str = "",
    ):
        losses: list[torch.Tensor] = []
        for target_name, head_config in self.hparams.properties.items():
            # Get the target and prediction
            prediction = predictions[target_name]
            label = labels[target_name]

            # Compute the loss
            loss = (
                self._compute_loss_for_head(head_config, prediction, label)
                * head_config.loss_coefficient
            )

            # Log the loss
            if log:
                self.log(f"{log_prefix}{target_name}_loss", loss)
            losses.append(loss)

        # Sum the losses
        loss = cast(torch.Tensor, sum(losses))

        # Log the total loss & return
        if log:
            self.log(f"{log_prefix}total_loss", loss)
        return loss

    def _common_step(
        self,
        batch: TBatch,
        name: str,
        metrics: FinetuneMetrics | None,
        log: bool = True,
    ):
        try:
            predictions = self(batch)
        except _SkipBatchError:

            def _zero_loss():
                # Return a zero loss tensor that is still attached to all
                #   parameters so that the optimizer can still update them.
                # This prevents DDP unused parameter errors.
                return cast(torch.Tensor, sum(p.sum() * 0.0 for p in self.parameters()))

            return _zero_loss()

        # Extract labels from the batch
        labels = self.batch_to_labels(
            batch,
            self.hparams.properties,
        )

        # Compute loss
        loss = self._compute_loss(
            predictions,
            labels,
            log=log,
            log_prefix=f"{name}/",
        )

        # Log metrics
        if log and metrics is not None:
            self.log_dict(
                {
                    f"{name}/{metric_name}": metric
                    for metric_name, metric in metrics(predictions, labels).items()
                }
            )

        return predictions, loss

    @override
    def training_step(self, batch: TBatch, batch_idx: int):
        _, loss = self._common_step(batch, "train", self.train_metrics)
        return loss

    @override
    def validation_step(self, batch: TBatch, batch_idx: int):
        _ = self._common_step(batch, "val", self.val_metrics)

    @override
    def test_step(self, batch: TBatch, batch_idx: int):
        _ = self._common_step(batch, "test", self.test_metrics)

    @override
    def predict_step(self, batch: TBatch, batch_idx: int):
        prediction, _ = self._common_step(batch, "predict", None, log=False)
        return prediction

    @override
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer.construct_optimizer(self.parameters())
        lr_scheduler = self.hparams.lr_scheduler.construct_lr_scheduler(optimizer)
        return cast(
            OptimizerLRSchedulerConfig,
            {"optimizer": optimizer, "lr_scheduler": lr_scheduler},
        )
