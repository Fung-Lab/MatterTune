from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import ase
import torch
from lightning.pytorch import Trainer
from torch.utils.data import Dataset
from typing_extensions import override

from ..finetune.properties import PropertyConfig

if TYPE_CHECKING:
    from ..finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig


log = logging.getLogger(__name__)


class MatterTunePropertyPredictor:
    """
    A wrapper class for handling predictions using a fine-tuned MatterTune model.

    This class provides an interface to make predictions using a trained neural network.
    It wraps a PyTorch Lightning module and handles the necessary setup for making
    predictions on atomic systems.

    lightning_module : FinetuneModuleBase
        The trained PyTorch Lightning module that will be used for predictions.
    lightning_trainer_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the PyTorch Lightning Trainer.
        Defaults to None.

    Examples
    --------
    >>> from mattertune.wrappers import MatterTunePropertyPredictor
    >>> predictor = MatterTunePropertyPredictor(trained_model)  # OR `predictor = trained_model.property_predictor()`
    >>> predictions = predictor.predict(atoms_list)

    The class provides a simplified interface for making predictions with trained models,
    handling the necessary setup of trainers and dataloaders internally.
    """

    def __init__(
        self,
        lightning_module: FinetuneModuleBase[Any, Any, FinetuneModuleBaseConfig],
        lightning_trainer_kwargs: dict[str, Any] | None = None,
    ):
        self.lightning_module = lightning_module
        self._lightning_trainer_kwargs = lightning_trainer_kwargs or {}

    def predict(
        self,
        atoms_list: list[ase.Atoms],
        properties: Sequence[str | PropertyConfig] | None = None,
        *,
        batch_size: int = 1,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Predicts properties for a list of atomic systems using the trained model.

        This method processes a list of atomic structures through the model and returns
        predicted properties for each system.

        Parameters
        ----------
        atoms_list : list[ase.Atoms]
            List of atomic systems to predict properties for.
        properties : Sequence[str | PropertyConfig] | None, optional
            Properties to predict. Can be specified as strings or PropertyConfig objects.
            If None, predicts all properties supported by the model.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            List of dictionaries containing predicted properties for each system.
            Each dictionary maps property names to torch.Tensor values.

        Notes
        -----
        - Creates a temporary trainer instance for prediction
        - Converts input atoms to a dataloader compatible with the model
        - Returns raw prediction outputs from the model
        """
        # Resolve `properties` to a list of `PropertyConfig` objects.
        properties = _resolve_properties(properties, self.lightning_module.hparams)

        # Create a trainer instance.
        trainer = _create_trainer(self._lightning_trainer_kwargs, self.lightning_module)

        # Create a dataloader from the atoms_list.
        dataloader = _atoms_list_to_dataloader(
            atoms_list, self.lightning_module, batch_size
        )

        # Make predictions using the trainer.
        predictions = trainer.predict(
            self.lightning_module, dataloader, return_predictions=True
        )
        assert predictions is not None, "Predictions should not be None. Report a bug."
        predictions = cast(list[dict[str, torch.Tensor]], predictions)

        all_predictions = []
        for batch_preds in predictions:
            first_tensor = next(iter(batch_preds.values()))
            batch_size = len(first_tensor)
            for idx in range(batch_size):
                pred_dict = {}
                for key, value in batch_preds.items():
                    pred_dict[key] = torch.tensor(value[idx])
                all_predictions.append(pred_dict)
        assert len(all_predictions) == len(
            atoms_list
        ), "Mismatch in predictions length."
        return all_predictions


def _resolve_properties(
    properties: Sequence[str | PropertyConfig] | None,
    hparams: FinetuneModuleBaseConfig,
):
    # If `None`, return all properties.
    if properties is None:
        return hparams.properties

    resolved_properties: list[PropertyConfig] = []
    for prop in properties:
        # If `PropertyConfig`, append it to the list.
        if not isinstance(prop, str):
            resolved_properties.append(prop)
            continue

        # If string, it must be present in the hparams.
        if (
            prop := next((p for p in hparams.properties if p.name == prop), None)
        ) is None:
            raise ValueError(
                f"Property '{prop}' not found in the model hyperparameters. "
                f"Available properties: {', '.join(p.name for p in hparams.properties)}"
            )
        resolved_properties.append(prop)

    return resolved_properties


def _create_trainer(
    trainer_kwargs: dict[str, Any],
    lightning_module: FinetuneModuleBase,
):
    # Resolve the full trainer kwargs
    trainer_kwargs_resolved: dict[str, Any] = {"barebones": True}
    if lightning_module.requires_disabled_inference_mode():
        if (
            user_inference_mode := trainer_kwargs.get("inference_mode")
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
        trainer_kwargs_resolved["inference_mode"] = False

    # Update with the user-specified kwargs
    trainer_kwargs_resolved.update(trainer_kwargs)

    # Create the trainer
    trainer = Trainer(**trainer_kwargs_resolved)
    return trainer


def _atoms_list_to_dataloader(
    atoms_list: list[ase.Atoms],
    lightning_module: FinetuneModuleBase,
    batch_size: int = 1,
):
    class AtomsDataset(Dataset):
        def __init__(self, atoms_list: list[ase.Atoms]):
            self.atoms_list = atoms_list

        def __len__(self):
            return len(self.atoms_list)

        @override
        def __getitem__(self, idx):
            return self.atoms_list[idx]

    dataset = AtomsDataset(atoms_list)
    dataloader = lightning_module.create_dataloader(
        dataset,
        has_labels=False,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader
