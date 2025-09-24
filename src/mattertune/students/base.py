from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Generic
from typing_extensions import TypeVar, Any, cast, override, Unpack
from collections.abc import Callable, Iterable, Mapping, Sequence

import ase
import nshconfig as C
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


from mattertune.finetune.base import TData, TBatch, ModelOutput, _SkipBatchError
from mattertune.finetune.metrics import FinetuneMetrics
from mattertune.finetune.properties import PropertyConfig
from mattertune.finetune.optimizer import OptimizerConfig, create_optimizer
from mattertune.finetune.lr_scheduler import LRSchedulerConfig, ReduceOnPlateauConfig, create_lr_scheduler
from mattertune.finetune.loss import compute_loss
from mattertune.finetune.loader import DataLoaderKwargs, create_dataloader

log = logging.getLogger(__name__)


class StudentModuleBaseConfig(C.Config, ABC):
    """
    Base class for student model configuration.
    """
    
    ignore_gpu_batch_transform_error: bool = True
    """Whether to ignore data processing errors during training."""
    
    properties: Sequence[PropertyConfig]
    """Properties to predict."""
    
    optimizer: OptimizerConfig
    """Optimizer."""

    lr_scheduler: LRSchedulerConfig | None = None
    """Learning Rate Scheduler"""
    
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

    @abstractmethod
    def create_model(self) -> StudentModuleBase:
        """
        Creates an instance of the finetune module for this configuration.
        """

TStudentModuleConfig = TypeVar(
    "TStudentModuleConfig",
    bound=StudentModuleBaseConfig,
    covariant=True,
)
    
class StudentModuleBase(
    LightningModule,
    ABC,
    Generic[TData, TBatch, TStudentModuleConfig],
):
    """
    Student module base class. Inherits ``lightning.pytorch.LightningModule``
    """
    
    @classmethod
    @abstractmethod
    def hparams_cls(cls) -> type[TStudentModuleConfig]:
        """
        Returns the class of the hyperparameters configuration.
        """
        ...
    
    @abstractmethod
    def requires_disabled_inference_mode(self) -> bool:
        """
        Whether the model requires inference mode to be disabled.
        Normally for a strictly conservative force field, this should be True to ensure gradients are computed.
        """
        ...
    
    @abstractmethod
    def create_model(self):
        """
        Initialize student model here, based on model configuration in hparams.
        
        You should also construct any other ``nn.Module`` instances
        necessary for the forward pass here.
        """
        ...
    
    @abstractmethod
    def model_forward_context(
        self, data: TBatch, mode: str
    ) -> contextlib.AbstractContextManager:
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
        mode: str,
    ) -> ModelOutput:
        """
        Forward pass of the model.

        Args:
            batch: Input batch.

        Returns:
            Prediction of the model.
        """
        ...
    
    @abstractmethod
    def cpu_data_transform(self, data: TData) -> TData:
        """
        Transform data (on the CPU) before being batched and sent to the GPU.
        This is useful for any preprocessing that needs to be done on the CPU,
        however, two prepocessing steps are normally not be done here:
        1. Graph building (normally done in the atoms_to_data)
        2. setting pos.requires_grad_() for gradient-based force prediction (normally done in model_forward_context)
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
    def batch_to_labels(self, batch: TBatch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Extract ground truth values from a batch. 
        
        Returns:
            labels: Dictionary of ground truth values.
            mask: Dictionary of masks for the ground truth values, each mask is an integer tensor of shape (batch_size,) or (num_atoms,) with 1 for valid values and 0 for misssing values.
            Both dictionaries should use the property names as keys.
            mask[key] should be a 1-dim tensor with the same length as labels[key].
        """
        ...
        
    @abstractmethod
    def batch_to_num_atoms(self, batch: TBatch) -> torch.Tensor:
        """
        Extract number of atoms in each structure from a batch.

        Returns:
            num_atoms: Tensor of shape (batch_size,) with the number of atoms in each structure.
        """
        ...
    
    @abstractmethod
    def atoms_to_data(self, atoms: ase.Atoms, has_labels: bool) -> TData:
        """
        Convert an ASE atoms object to a data object. This is used to convert
        the input data to the format expected by the model.

        Args:
            atoms: ASE atoms object.
            has_labels: Whether the atoms object contains labels.
        """
        ...
    

    # endregion

    hparams: TStudentModuleConfig  # pyright: ignore[reportIncompatibleMethodOverride]
    hparams_initial: TStudentModuleConfig # pyright: ignore[reportIncompatibleMethodOverride]
    
    def __init__(self, hparams: TStudentModuleConfig | Mapping[str, Any]):
        hparams_cls = self.hparams_cls()
        if not isinstance(hparams, hparams_cls):
            hparams = hparams_cls.model_validate(hparams)
            
        super().__init__()
        
        # save hyperparameters
        self.save_hyperparameters(hparams)
        
        # create model
        self.create_model()
        
        # Create metrics
        self.create_metrics()
        
    def create_metrics(self):
        self.train_metrics = FinetuneMetrics(self.hparams.properties)
        self.val_metrics = FinetuneMetrics(self.hparams.properties)
        self.test_metrics = FinetuneMetrics(self.hparams.properties)
    
    @override
    def forward(
        self,
        batch: TBatch,
        mode: str,
        ignore_gpu_batch_transform_error: bool | None = None,
    ) -> ModelOutput:
        if ignore_gpu_batch_transform_error is None:
            ignore_gpu_batch_transform_error = (
                self.hparams.ignore_gpu_batch_transform_error
            )

        with self.model_forward_context(batch, mode):
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
            model_output = self.model_forward(
                batch, mode=mode
            )

            model_output["predicted_properties"] = {
                prop_name: prop_value.contiguous()
                for prop_name, prop_value in model_output[
                    "predicted_properties"
                ].items()
            }

            return model_output
        
    def _compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        label_masks: dict[str, torch.Tensor],
        log: bool = True,
        log_prefix: str = "",
    ):
        losses: list[torch.Tensor] = []
        for prop in self.hparams.properties:
            # Get the target and prediction
            prediction = predictions[prop.name]
            label = labels[prop.name]
            
            if label_masks is not None and prop.name in label_masks:
                mask = label_masks[prop.name].bool()
                label = label[mask]
                prediction = prediction[mask]

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
        mode: str,
        metrics: FinetuneMetrics | None,
        log: bool = True,
    ):
        try:
            output: ModelOutput = self(batch, mode=mode)
        except _SkipBatchError:

            def _zero_output():
                return {
                    "predicted_properties": {},
                }

            def _zero_loss():
                # Return a zero loss tensor that is still attached to all
                #   parameters so that the optimizer can still update them.
                # This prevents DDP unused parameter errors.
                return cast(torch.Tensor, sum(p.sum() * 0.0 for p in self.parameters()))

            return _zero_output(), _zero_loss()

        # Extract labels from the batch
        labels, label_masks = self.batch_to_labels(batch)
        predictions = output["predicted_properties"]

        for key, value in labels.items():
            labels[key] = value.contiguous()
            label_masks[key] = label_masks[key].contiguous()

        # Compute loss
        loss = self._compute_loss(
            predictions,
            labels,
            label_masks,
            log=log,
            log_prefix=f"{mode}/",
        )

        # Log metrics
        if log and (metrics is not None):
            log_metrics = {
                f"{mode}/{metric_name}": metric
                for metric_name, metric in metrics(predictions, labels).items()
            }
            self.log_dict(
                log_metrics,
                on_epoch=True,
                sync_dist=True,
            )
        return output, loss

    @override
    def training_step(self, batch: TBatch, batch_idx: int):
        _, loss = self._common_step(
            batch,
            "train",
            self.train_metrics,
        )
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    @override
    def validation_step(self, batch: TBatch, batch_idx: int):
        _ = self._common_step(batch, "val", self.val_metrics)

    @override
    def test_step(self, batch: TBatch, batch_idx: int):
        _ = self._common_step(batch, "test", self.test_metrics)

    @override
    def predict_step(self, batch: TBatch, batch_idx: int) -> list[dict[str, torch.Tensor]]:
        output: ModelOutput = self(
            batch, mode="predict", ignore_gpu_batch_transform_error=False
        )
        predictions = output["predicted_properties"]
        num_atoms = self.batch_to_num_atoms(batch).detach().cpu()
        pred_list = []
        for i in range(len(num_atoms)):
            pred_dict = {}
            for key, value in predictions.items():
                value = value.detach().cpu()
                if key == "energies_per_atom":
                    prop_type = "atom"
                else:
                    assert (
                        prop := next(
                            (p for p in self.hparams.properties if p.name == key), None
                        )
                    ) is not None, (
                        f"Property {key} not found in properties. "
                        "This should not happen, please report this."
                    )
                    prop_type = prop.property_type()
                match prop_type:
                    case "atom":
                        pred_dict[key] = value[torch.sum(num_atoms[:i]):torch.sum(num_atoms[:i])+num_atoms[i]]
                    case "system":
                        pred_dict[key] = value[i]
                    case _:
                        raise ValueError(f"Unknown property type: {prop_type}")
            pred_list.append(pred_dict)
        return pred_list

    def trainable_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        return self.named_parameters()

    @override
    def configure_optimizers(self):
        optimizer = create_optimizer(
            self.hparams.optimizer, self.trainable_parameters()
        )
        return_config: OptimizerLRSchedulerConfig = {"optimizer": optimizer} # type: ignore

        if (lr_scheduler := self.hparams.lr_scheduler) is not None:
            scheduler_class = create_lr_scheduler(lr_scheduler, optimizer)
            if isinstance(lr_scheduler, ReduceOnPlateauConfig):
                return_config["lr_scheduler"] = {
                    "scheduler": scheduler_class,
                    "monitor": lr_scheduler.monitor,
                }
            else:
                return_config["lr_scheduler"] = scheduler_class
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

    def property_predictor(
        self, lightning_trainer_kwargs: dict[str, Any] | None = None
    ):
        """Return a wrapper for easy prediction without explicitly setting up a lightning trainer.

        This method provides a high-level interface for making predictions with the trained model.

        This can be used for various prediction tasks including but not limited to:
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
        MatterTunePropertyPredictor
            A wrapper class that provides simplified prediction functionality without requiring
            direct interaction with the Lightning Trainer.
        Examples
        --------
        >>> model = MyModel()
        >>> property_predictor = model.property_predictor()
        >>> atoms_1 = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True)
        >>> atoms_2 = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True)
        >>> atoms = [atoms_1, atoms_2]
        >>> predictions = property_predictor.predict(atoms, ["energy", "forces"])
        >>> print("Atoms 1 energy:", predictions[0]["energy"])
        >>> print("Atoms 1 forces:", predictions[0]["forces"])
        """

        from ..wrappers.property_predictor import MatterTunePropertyPredictor

        return MatterTunePropertyPredictor(
            self,
            lightning_trainer_kwargs=lightning_trainer_kwargs,
        )

    def ase_calculator(
        self, 
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        """Returns an ASE calculator wrapper for the interatomic potential.

        This method creates an ASE (Atomic Simulation Environment) calculator that can be used
        to compute energies and forces using the trained interatomic potential model.

        The calculator integrates with ASE's standard interfaces for molecular dynamics
        and structure optimization.

        Parameters
        ----------
        device : str, optional

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

        return MatterTuneCalculator(self, device=torch.device(device))