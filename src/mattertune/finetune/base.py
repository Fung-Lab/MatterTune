from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Generic, Literal

import ase
import nshconfig as C
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.utils.data import Dataset
from typing_extensions import NotRequired, TypedDict, TypeVar, Unpack, cast, override

from ..normalization import ComposeNormalizers, NormalizationContext, NormalizerConfig
from .loader import DataLoaderKwargs, create_dataloader
from .loss import compute_loss
from .lr_scheduler import LRSchedulerConfig, ReduceOnPlateauConfig, create_lr_scheduler
from .metrics import FinetuneMetrics
from .optimizer import OptimizerConfig, create_optimizer
from .properties import PropertyConfig

log = logging.getLogger(__name__)


class FinetuneModuleBaseConfig(C.Config, ABC):
    
    reset_backbone: bool = False
    """Whether to reset the backbone of the model when creating the model."""
    
    reset_output_heads: bool = False
    """Whether to reset the output heads of the model when creating the model."""

    properties: Sequence[PropertyConfig]
    """Properties to predict."""

    optimizer: OptimizerConfig
    """Optimizer."""

    lr_scheduler: LRSchedulerConfig | None = None
    """Learning Rate Scheduler"""

    ignore_gpu_batch_transform_error: bool = True
    """Whether to ignore data processing errors during training."""

    normalizers: Mapping[str, Sequence[NormalizerConfig]] = {}
    """Normalizers for the properties.

    Any property can be associated with multiple normalizers. This is useful
    for cases where we want to normalize the same property in different ways.
    For example, we may want to normalize the energy by subtracting
    the atomic reference energies, as well as by mean and standard deviation
    normalization.

    The normalizers are applied in the order they are defined in the list.
    """
    
    early_stop_message_passing: int | None = None
    """Number of message passing steps for early stopping. If None, no early stopping is applied."""
    
    using_partition: bool = False
    """Whether to be using partitioning in the model."""

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
    def create_model(self) -> FinetuneModuleBase:
        """
        Creates an instance of the finetune module for this configuration.
        """

    @override
    def __post_init__(self):
        # VALIDATION: Any key for `normalizers` or `referencers` should be a property name.
        for key in self.normalizers.keys():
            property_names = [prop.name for prop in self.properties]
            if key not in property_names:
                raise ValueError(
                    f"Key '{key}' in 'normalizers' is not a valid property name."
                )


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
R = TypeVar("R", infer_variance=True)


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
    def hparams_cls(cls) -> type[TFinetuneModuleConfig]:
        """Return the hyperparameters config class for this module."""
        ...

    # region ABC methods for output heads and model forward pass
    @abstractmethod
    def create_model(self):
        """
        Initialize both the pre-trained backbone and the output heads for the properties to predict.

        You should also construct any other ``nn.Module`` instances
        necessary for the forward pass here.
        """
        ...
        
    @abstractmethod
    def apply_early_stop_message_passing(self, message_passing_steps: int|None):
        """
        Apply message passing for early stopping.
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
    def requires_disabled_inference_mode(self) -> bool:
        """
        Whether the model requires inference mode to be disabled.
        """
        ...

    @abstractmethod
    def model_forward(
        self,
        batch: TBatch,
        mode: str,
        return_backbone_output: bool = False,
        using_partition: bool = False,
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
    def apply_callable_to_backbone(self, fn: Callable[[nn.Module], R]) -> R:
        """
        Apply a callable to the backbone model and return the result.

        This is useful for applying functions to the backbone model that are not
        part of the standard forward pass. For example, this can be used to
        update structure or weights of the backbone model, e.g., for LoRA.

        Args:
            fn: Callable to apply to the backbone model.

        Returns:
            Result of the callable.
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
    def atoms_to_data(self, atoms: ase.Atoms, has_labels: bool) -> TData:
        """
        Convert an ASE atoms object to a data object. This is used to convert
        the input data to the format expected by the model.

        Args:
            atoms: ASE atoms object.
            has_labels: Whether the atoms object contains labels.
        """
        ...
        
    @abstractmethod
    def get_connectivity_from_data(self, data: TData) -> torch.Tensor:
        """
        Get the connectivity from the data. This is used to extract the connectivity
        information from the data object. This is useful for message passing
        and other graph-based operations.
        
        Returns:
            edge_index: Tensor of shape (2, num_edges) containing the src and dst indices of the edges.
        """
        ...
        
    @abstractmethod
    def get_connectivity_from_atoms(self, atoms: ase.Atoms) -> torch.Tensor:
        """
        Get the connectivity from the data. This is used to extract the connectivity
        information from the data object. This is useful for message passing
        and other graph-based operations.
        
        Returns:
            edge_index: Tensor of shape (2, num_edges) containing the src and dst indices of the edges.
        """
        ...

    @abstractmethod
    def create_normalization_context_from_batch(
        self, batch: TBatch
    ) -> NormalizationContext:
        """
        Create a normalization context from a batch. This is used to normalize
        and denormalize the properties.

        The normalization context contains all the information required to
        normalize and denormalize the properties. Currently, this only
        includes the compositions of the materials in the batch.
        The compositions should be provided as an integer tensor of shape
        (batch_size, num_elements), where each row (i.e., `compositions[i]`)
        corresponds to the composition vector of the `i`-th material in the batch.

        The composition vector is a vector that maps each element to the number of
        atoms of that element in the material. For example, `compositions[:, 1]`
        corresponds to the number of Hydrogen atoms in each material in the batch,
        `compositions[:, 2]` corresponds to the number of Helium atoms, and so on.

        Args:
            batch: Input batch.

        Returns:
            Normalization context.
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
        self.apply_early_stop_message_passing(self.hparams.early_stop_message_passing)
        
        if self.hparams.reset_backbone:
            for name, param in self.backbone.named_parameters():
                if param.dim() > 1:
                    print(f"Resetting {name}")
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

        # Create metrics
        self.create_metrics()

        # Create normalization modules
        self.create_normalizers()

        # Ensure that some parameters require gradients
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError(
                "No parameters require gradients. "
                "Please ensure that some parts of the model are trainable."
            )
            
        self.predict_mode = "property"
        self.diabled_heads = []
        
    def set_disabled_heads(self, disabled_heads: list[str]):
        self.disabled_heads = disabled_heads

    def create_metrics(self):
        self.train_metrics = FinetuneMetrics(self.hparams.properties)
        self.val_metrics = FinetuneMetrics(self.hparams.properties)
        self.test_metrics = FinetuneMetrics(self.hparams.properties)

    def create_normalizers(self):
        self.normalizers = nn.ModuleDict(
            {
                prop.name: ComposeNormalizers(
                    [
                        normalizer.create_normalizer_module()
                        for normalizer in normalizers
                    ]
                )
                for prop in self.hparams.properties
                if (normalizers := self.hparams.normalizers.get(prop.name))
            }
        )

    def normalize(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        ctx: NormalizationContext,
    ):
        """
        Normalizes predictions and targets

        Args:
            predictions: Dictionary of predicted values to normalize. 
            targets: Dictionary of target values to normalize. 
            ctx: Normalization context. This should be created using
                ``create_normalization_context_from_batch``.

        Returns:
            Normalized predictions and targets.
        """
        normalized_predictions = {}
        normalized_targets = {}
        for key in predictions.keys():
            pred = predictions[key]
            target = None if targets is None else targets[key]
            if key in self.normalizers:
                normalizer = cast(ComposeNormalizers, self.normalizers[key])
                pred, target = normalizer.normalize(predictions[key], targets[key], ctx)
            normalized_predictions[key] = pred
            normalized_targets[key] = target
        return normalized_predictions, normalized_targets

    def denormalize(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        ctx: NormalizationContext,
    ):
        """
        Denormalizes predictions and targets

        Args:
            predictions: Dictionary of predicted values to denormalize.
            targets: Dictionary of target values to denormalize.
            ctx: Normalization context. This should be created using
                ``create_normalization_context_from_batch``.

        Returns:
            Denormalized predictions and targets.
        """
        denormalized_predictions = {}
        denormalized_targets = {}
        for key in predictions.keys():
            pred = predictions[key]
            target = targets[key]
            if key in self.normalizers:
                normalizer = cast(ComposeNormalizers, self.normalizers[key])
                pred, target = normalizer.denormalize(pred, target, ctx)
            denormalized_predictions[key] = pred
            denormalized_targets[key] = target
        return denormalized_predictions, denormalized_targets
    
    def denormalize_predict(
        self,
        predictions: dict[str, torch.Tensor],
        ctx: NormalizationContext,
    ):
        """
        Denormalizes predictions

        Args:
            predictions: Dictionary of predicted values to denormalize.
            ctx: Normalization context. This should be created using
                ``create_normalization_context_from_batch``.

        Returns:
            Denormalized predictions.
        """
        denormalized_predictions = {}
        for key in predictions.keys():
            pred = predictions[key]
            if key in self.normalizers:
                normalizer = cast(ComposeNormalizers, self.normalizers[key])
                pred = normalizer.denormalize_predict(pred, ctx)
            denormalized_predictions[key] = pred
        return denormalized_predictions

    @override
    def forward(
        self,
        batch: TBatch,
        mode: str,
        return_backbone_output: bool = False,
        ignore_gpu_batch_transform_error: bool | None = None,
        using_partition: bool = False,
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
                batch, mode=mode, return_backbone_output=return_backbone_output, using_partition=using_partition
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
                    "backbone_output": None,
                }

            def _zero_loss():
                # Return a zero loss tensor that is still attached to all
                #   parameters so that the optimizer can still update them.
                # This prevents DDP unused parameter errors.
                return cast(torch.Tensor, sum(p.sum() * 0.0 for p in self.parameters()))

            return _zero_output(), _zero_loss()

        # Extract labels from the batch
        labels = self.batch_to_labels(batch)
        predictions = output["predicted_properties"]

        if len(self.normalizers) > 0:
            # Create the normalization context required for normalization/referencing.
            # We only need to create the context once per batch.
            normalization_ctx = self.create_normalization_context_from_batch(batch)
            predictions, labels = self.normalize(predictions, labels, normalization_ctx)

        for key, value in labels.items():
            labels[key] = value.contiguous()

        # Compute loss
        loss = self._compute_loss(
            predictions,
            labels,
            log=log,
            log_prefix=f"{mode}/",
        )
        
        # NOTE: After computing the loss, we denormalize the predictions.
        if len(self.normalizers) > 0:
            predictions, labels = self.denormalize(predictions, labels, normalization_ctx) # type: ignore

        # Log metrics
        if log and (metrics is not None):
            denormalized_metrics = {
                f"{mode}/{metric_name}": metric
                for metric_name, metric in metrics(predictions, labels).items()
            }
            self.log_dict(
                denormalized_metrics,
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
    def predict_step(self, batch: TBatch, batch_idx: int):
        if self.predict_mode == "property":
            output: ModelOutput = self(
                batch, mode="predict", ignore_gpu_batch_transform_error=False, using_partition=self.hparams.using_partition
            )
            predictions = output["predicted_properties"]
            normalization_ctx = self.create_normalization_context_from_batch(batch)
            if len(self.normalizers) > 0:
                predictions = self.denormalize_predict(predictions, normalization_ctx)
            ## split predictions into a list of dicts
            num_atoms = normalization_ctx.num_atoms
            pred_list = []
            for i in range(len(num_atoms)):
                pred_dict = {}
                for key, value in predictions.items():
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
        elif self.predict_mode == "internal_feature":
            output: ModelOutput = self(
                batch,
                mode="predict",
                ignore_gpu_batch_transform_error=False,
                return_backbone_output=True,
            )
            assert "backbone_output" in output
            backbone_output = output["backbone_output"]
            for key, value in backbone_output.items():
                if isinstance(value, torch.Tensor):
                    backbone_output[key] = value.detach().cpu()
            return backbone_output

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
        
    def internal_feature_predictor(
        self, lightning_trainer_kwargs: dict[str, Any] | None = None
    ):
        from ..wrappers.property_predictor import MatterTuneInternalFeaturePredictor
        
        return MatterTuneInternalFeaturePredictor(
            self,
            lightning_trainer_kwargs=lightning_trainer_kwargs,
        )

    def ase_calculator(
        self, 
        # lightning_trainer_kwargs: dict[str, Any] | None = None,
        device: str = "cpu",
    ):
        from ..wrappers.ase_calculator import MatterTuneCalculator
        
        return MatterTuneCalculator(self, device=torch.device(device))
    
    def ase_calculator_with_partition(
        self, 
        mp_steps: int,
        num_partitions: int,
        batch_size: int = 1,
        lightning_trainer_kwargs: dict[str, Any] = {},
    ):
        from ..wrappers.ase_calculator import MatterTunePartitionCalculator
        
        return MatterTunePartitionCalculator(
            self,
            mp_steps=mp_steps,
            num_partitions=num_partitions,
            batch_size=batch_size,
            lightning_trainer_kwargs=lightning_trainer_kwargs,
        )