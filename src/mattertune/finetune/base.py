from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic

import ase
import nshconfig as C
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.utils.data import Dataset
from typing_extensions import NotRequired, TypedDict, TypeVar, Unpack, cast, override

from .loader import DataLoaderKwargs, create_dataloader
from .loss import compute_loss
from .lr_scheduler import LRSchedulerConfig, create_lr_scheduler
from .metrics import FinetuneMetrics
from .optimizer import OptimizerConfig, create_optimizer
from .properties import PropertyConfig

if TYPE_CHECKING:
    from ase import Atoms

log = logging.getLogger(__name__)


class FinetuneModuleBaseConfig(C.Config, ABC):
    properties: Sequence[PropertyConfig]
    """Properties to predict."""

    optimizer: OptimizerConfig
    """Optimizer."""

    lr_scheduler: LRSchedulerConfig | None = None
    """Learning Rate Scheduler"""

    ignore_gpu_batch_transform_error: bool = True
    """Whether to ignore data processing errors during training."""

    @classmethod
    @abstractmethod
    def model_cls(cls) -> type[FinetuneModuleBase]: ...


class _SkipBatchError(Exception):
    """
    Exception to skip a batch in the forward pass. This is not a real error and
        should not be logged.

    Instead, this is basically a control flow mechanism to skip a batch
        if an error occurs during the forward pass. This is useful for
        handling edge cases where a batch may be invalid or cause an error
        during the forward pass. In this case, we can throw this exception
        anywhere in the forward pas and then catch it in the `_common_step`
        method. If this exception is caught, we can just skip the batch
        instead of logging an error.

    This is primarily used to skip graph generation errors in messy data. E.g.,
        if our dataset contains materials with so little atoms that we cannot
        generate a graph, we can just skip these materials instead of
        completely failing the training run.
    """

    pass


class ModelOutput(TypedDict):
    predicted_properties: dict[str, torch.Tensor]
    """Predicted properties. This dictionary should be exactly
        in the same shape/format  as the output of `batch_to_labels`."""

    backbone_output: NotRequired[Any]
    """Output of the backbone model. Only set if `return_backbone_output` is True."""


TData = TypeVar("TData")
TBatch = TypeVar("TBatch")
TFinetuneModuleConfig = TypeVar(
    "TFinetuneModuleConfig",
    bound=FinetuneModuleBaseConfig,
    covariant=True,
)


class FinetuneModuleBase(
    LightningModule,
    ABC,
    Generic[TData, TBatch, TFinetuneModuleConfig],
):
    """
    Finetune module base class. Inherits ``lightning.pytorch.LightningModule``.
    """

    @classmethod
    @abstractmethod
    def hparams_cls(cls) -> type[FinetuneModuleBaseConfig]:
        """Return the hyperparameters config class for this module."""
        ...

    @classmethod
    @abstractmethod
    def ensure_dependencies(cls):
        """
        Ensure that all dependencies are installed.

        This method should raise an exception if any dependencies are missing,
            with a message indicating which dependencies are missing and
            how to install them.
        """
        ...

    # region ABC methods for output heads and model forward pass
    @abstractmethod
    def create_model(self):
        """
        Initialize both the pre-trained backbone and the
            output heads for the properties to predict.

        You should also construct any other ``nn.Module`` instances
            necessary for the forward pass here.
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
    def requires_disabled_inference_mode(self) -> bool:
        """
        Whether the model requires inference mode to be disabled.
        """
        ...

    @abstractmethod
    def model_forward(
        self,
        batch: TBatch,
        return_backbone_output: bool = False,
    ) -> ModelOutput:
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
        """
        ...

    @abstractmethod
    def output_head_parameters(self) -> Iterable[nn.Parameter]:
        """
        Return the parameters of the output heads.
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

        This will mainly be used to compute the (radius or knn) graph from
            the atomic positions.
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
    def atoms_to_data(self, atoms: Atoms, has_labels: bool) -> TData:
        """
        Convert an ASE atoms object to a data object. This is used to convert
            the input data to the format expected by the model.

        Args:
            atoms: ASE atoms object.
            has_labels: Whether the atoms object contains labels.
        """
        ...

    # endregion

    hparams: TFinetuneModuleConfig  # pyright: ignore[reportIncompatibleMethodOverride]
    hparams_initial: TFinetuneModuleConfig  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, hparams: TFinetuneModuleConfig | Mapping[str, Any]):
        hparams_cls = self.hparams_cls()
        if not isinstance(hparams, hparams_cls):
            hparams = hparams_cls.model_validate(hparams)

        super().__init__()

        # Save the hyperparameters
        self.save_hyperparameters(hparams)

        # Create the backbone model and output heads
        self.create_model()

        # Create metrics
        self.create_metrics()

        # Ensure that some parameters require gradients
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError(
                "No parameters require gradients. "
                "Please ensure that some parts of the model are trainable."
            )

    def create_metrics(self):
        self.train_metrics = FinetuneMetrics(self.hparams.properties)
        self.val_metrics = FinetuneMetrics(self.hparams.properties)
        self.test_metrics = FinetuneMetrics(self.hparams.properties)

    @override
    def forward(
        self,
        batch: TBatch,
        return_backbone_output: bool = False,
        ignore_gpu_batch_transform_error: bool | None = None,
    ) -> ModelOutput:
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

    def _compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        log: bool = True,
        log_prefix: str = "",
    ):
        losses: list[torch.Tensor] = []
        for prop in self.hparams.properties:
            # Get the target and prediction
            prediction = predictions[prop.name]
            label = labels[prop.name]

            # Compute the loss
            loss = compute_loss(prop.loss, prediction, label) * prop.loss_coefficient

            # Log the loss
            if log:
                self.log(f"{log_prefix}{prop.name}_loss", loss)
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
            output: ModelOutput = self(batch)
        except _SkipBatchError:

            def _zero_loss():
                # Return a zero loss tensor that is still attached to all
                #   parameters so that the optimizer can still update them.
                # This prevents DDP unused parameter errors.
                return cast(torch.Tensor, sum(p.sum() * 0.0 for p in self.parameters()))

            return _zero_loss()

        # Extract labels from the batch
        labels = self.batch_to_labels(batch)
        predictions = output["predicted_properties"]

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

        return output, loss

    @override
    def training_step(self, batch: TBatch, batch_idx: int):
        _, loss = self._common_step(
            batch,
            "train",
            self.train_metrics,
        )
        return loss

    @override
    def validation_step(self, batch: TBatch, batch_idx: int):
        _ = self._common_step(batch, "val", self.val_metrics)

    @override
    def test_step(self, batch: TBatch, batch_idx: int):
        _ = self._common_step(batch, "test", self.test_metrics)

    @override
    def predict_step(self, batch: TBatch, batch_idx: int):
        output: ModelOutput = self(batch, ignore_gpu_batch_transform_error=False)
        return output["predicted_properties"]

    @override
    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optimizer, self.parameters())
        return_config: OptimizerLRSchedulerConfig = {"optimizer": optimizer}

        if (lr_scheduler := self.hparams.lr_scheduler) is not None:
            return_config["lr_scheduler"] = create_lr_scheduler(lr_scheduler, optimizer)

        return return_config

    def create_dataloader(
        self,
        dataset: Dataset[ase.Atoms],
        has_labels: bool,
        **kwargs: Unpack[DataLoaderKwargs],
    ):
        """
        Creates a wrapped DataLoader for the given dataset.
        This will wrap the dataset with the CPU data transform and the model's
            collate function.

        NOTE about `has_labels`: This is used to determine whether our data
            loading pipeline should expect labels in the dataset. This should
            be `True` for train/val/test datasets (as we compute loss and metrics
            on these datasets) and `False` for prediction datasets.

        Args:
            dataset: Dataset to wrap.
            has_labels: Whether the dataset contains labels. This should be
                `True` for train/val/test datasets and `False` for prediction datasets.
            **kwargs: Additional keyword arguments to pass to the DataLoader.
        """
        return create_dataloader(dataset, has_labels, lightning_module=self, **kwargs)

    def potential(self, lightning_trainer_kwargs: dict[str, Any] | None = None):
        """Return a wrapper for easy prediction without explicitly setting up a lightning trainer.
        This method provides a high-level interface for making predictions with the trained model.
        While named 'potential', it can be used for various prediction tasks including but not
        limited to:
        - Interatomic potential energy and forces
        - Material property prediction
        - Structure-property relationships
        Parameters
        ----------
        lightning_trainer_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments to pass to the PyTorch Lightning Trainer.
            If None, default trainer settings will be used.
        Returns
        -------
        MatterTunePotential
            A wrapper class that provides simplified prediction functionality without requiring
            direct interaction with the Lightning Trainer.
        Examples
        --------
        >>> model = MyModel()
        >>> potential = model.potential()
        >>> atoms_1 = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True)
        >>> atoms_2 = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True)
        >>> atoms = [atoms_1, atoms_2]
        >>> predictions = potential.predict(atoms, ["energy", "forces"])
        >>> print("Atoms 1 energy:", predictions[0]["energy"])
        >>> print("Atoms 1 forces:", predictions[0]["forces"])
        >>> print("Atoms 2 energy:", predictions[1]["energy"])
        >>> print("Atoms 2 forces:", predictions[1]["forces"])
        """

        from ..wrappers.potential import MatterTunePotential

        return MatterTunePotential(
            self,
            lightning_trainer_kwargs=lightning_trainer_kwargs,
        )

    def ase_calculator(self, lightning_trainer_kwargs: dict[str, Any] | None = None):
        """Returns an ASE calculator wrapper for the interatomic potential.

        This method creates an ASE (Atomic Simulation Environment) calculator that can be used
        to compute energies and forces using the trained interatomic potential model.
        The calculator integrates with ASE's standard interfaces for molecular dynamics
        and structure optimization.

        Parameters
        ----------
        lightning_trainer_kwargs : dict[str, Any] | None, optional
            Keyword arguments to pass to the PyTorch Lightning trainer used for inference.
            If None, default trainer settings will be used.

        Returns
        -------
        MatterTuneCalculator
            An ASE calculator wrapper around the trained potential that can be used
            for energy and force calculations via ASE's interfaces.

        Examples
        --------
        >>> model = MyModel()
        >>> calc = model.ase_calculator()
        >>> atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True)
        >>> atoms.calc = calc
        >>> energy = atoms.get_potential_energy()
        >>> forces = atoms.get_forces()
        """
        from ..wrappers.ase_calculator import MatterTuneCalculator

        potential = self.potential(lightning_trainer_kwargs=lightning_trainer_kwargs)
        return MatterTuneCalculator(potential)
