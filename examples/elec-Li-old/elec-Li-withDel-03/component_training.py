from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np
import torch

from mattertune.finetune.base import _SkipBatchError
from mattertune.finetune.loss import compute_loss

from component_reference import FEATURE_NAMES, component_feature_vector


def _zero_loss(module: Any) -> torch.Tensor:
    return sum(parameter.sum() * 0.0 for parameter in module.parameters())


def attach_component_features(module: Any, feature_names: list[str] = FEATURE_NAMES) -> None:
    original_atoms_to_data = module.atoms_to_data
    feature_indices = [FEATURE_NAMES.index(name) for name in feature_names]

    def atoms_to_data_with_components(self: Any, atoms: Any, has_labels: bool):
        data = original_atoms_to_data(atoms, has_labels)
        features = component_feature_vector(atoms)[feature_indices]
        data.component_features = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
        return data

    module.atoms_to_data = MethodType(atoms_to_data_with_components, module)


def attach_component_reference(
    module: Any,
    *,
    feature_names: list[str],
    coefficients: np.ndarray,
    per_atom_energy_normalize: bool,
) -> None:
    attach_component_features(module, feature_names)
    coeff_tensor = torch.tensor(coefficients, dtype=torch.float32)

    def component_reference(batch: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coeff = coeff_tensor.to(device=device, dtype=dtype)
        features = batch.component_features.to(device=device, dtype=dtype)
        return (features @ coeff).reshape(-1)

    def _common_step_component(
        self: Any,
        batch: Any,
        mode: str,
        metrics: Any | None,
        log: bool = True,
    ):
        labels = self.batch_to_labels(batch)
        try:
            output = self(batch, mode=mode)
        except _SkipBatchError:
            return {"predicted_properties": {}}, _zero_loss(self)

        predictions = output["predicted_properties"]
        energy_prediction = predictions["energy"]
        energy_target = labels["energy"]
        reference = component_reference(batch, energy_prediction.device, energy_prediction.dtype)

        loss_predictions = dict(predictions)
        loss_labels = dict(labels)
        loss_labels["energy"] = energy_target - reference

        if per_atom_energy_normalize:
            normalization_ctx = self.create_normalization_context_from_batch(batch)
            num_atoms = normalization_ctx.num_atoms.to(
                device=energy_prediction.device,
                dtype=energy_prediction.dtype,
            )
            loss_predictions["energy"] = loss_predictions["energy"] / num_atoms
            loss_labels["energy"] = loss_labels["energy"] / num_atoms

        for key, value in loss_labels.items():
            loss_labels[key] = value.contiguous()

        losses: list[torch.Tensor] = []
        for prop in self.hparams.properties:
            loss = compute_loss(prop.loss, loss_predictions[prop.name], loss_labels[prop.name])
            loss = loss * prop.loss_coefficient
            losses.append(loss)
            if log:
                self.log(f"{mode}/{prop.name}_loss", loss)

        total_loss = sum(losses)
        if log:
            self.log(f"{mode}/total_loss", total_loss)

        denorm_predictions = dict(predictions)
        denorm_predictions["energy"] = energy_prediction + reference
        denorm_labels = labels

        if log and metrics is not None:
            self.log_dict(
                {
                    f"{mode}/{metric_name}": metric
                    for metric_name, metric in metrics(denorm_predictions, denorm_labels).items()
                },
                on_epoch=True,
                sync_dist=True,
            )

        return output, total_loss

    def predict_step_component(self: Any, batch: Any, batch_idx: int):
        output = self(batch, mode="predict", ignore_gpu_batch_transform_error=False)
        predictions = output["predicted_properties"]
        reference = component_reference(
            batch,
            predictions["energy"].device,
            predictions["energy"].dtype,
        )
        predictions = dict(predictions)
        predictions["energy"] = predictions["energy"] + reference

        normalization_ctx = self.create_normalization_context_from_batch(batch)
        num_atoms = normalization_ctx.num_atoms
        pred_list = []
        for index in range(len(num_atoms)):
            pred_dict = {}
            atom_start = torch.sum(num_atoms[:index])
            atom_stop = atom_start + num_atoms[index]
            for key, value in predictions.items():
                value = value.detach().cpu()
                if key == "forces":
                    pred_dict[key] = value[atom_start:atom_stop]
                else:
                    pred_dict[key] = value[index]
            pred_list.append(pred_dict)
        return pred_list

    module._common_step = MethodType(_common_step_component, module)
    module.predict_step = MethodType(predict_step_component, module)
