from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from typing_extensions import override

if TYPE_CHECKING:
    from ..finetune.properties import PropertyConfig
    from .property_predictor import MatterTunePropertyPredictor
    from ..finetune.base import FinetuneModuleBase


class MatterTuneCalculator(Calculator):
    """
    A fast version of the MatterTuneCalculator that uses the `predict_step` method directly without creating a trainer.
    """
    
    @override
    def __init__(self, model: FinetuneModuleBase, device: torch.device):
        super().__init__()

        self.model = model.to(device)

        self.implemented_properties: list[str] = []
        self._ase_prop_to_config: dict[str, PropertyConfig] = {}

        for prop in self.model.hparams.properties:
            # Ignore properties not marked as ASE calculator properties.
            if (ase_prop_name := prop.ase_calculator_property_name()) is None:
                continue
            self.implemented_properties.append(ase_prop_name)
            self._ase_prop_to_config[ase_prop_name] = prop

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ):
        if properties is None:
            properties = copy.deepcopy(self.implemented_properties)

        # Call the parent class to set `self.atoms`.
        Calculator.calculate(self, atoms)

        # Make sure `self.atoms` is set.
        assert self.atoms is not None, (
            "`MatterTuneCalculator.atoms` is not set. "
            "This should have been set by the parent class. "
            "Please report this as a bug."
        )
        assert isinstance(self.atoms, Atoms), (
            "`MatterTuneCalculator.atoms` is not an `ase.Atoms` object. "
            "This should have been set by the parent class. "
            "Please report this as a bug."
        )
        
        prop_configs = [self._ase_prop_to_config[prop] for prop in properties]
        
        normalized_atoms = copy.deepcopy(self.atoms)
        # scaled_pos = normalized_atoms.get_scaled_positions()
        # scaled_pos = np.mod(scaled_pos, 1.0)
        # normalized_atoms.set_scaled_positions(scaled_pos)
        
        batch = self.model.atoms_to_data(normalized_atoms, has_labels=False)
        batch = self.model.collate_fn([batch])
        batch = self.model.batch_to_device(batch, self.model.device)
        
        pred = self.model.predict_step(
            batch = batch,
            batch_idx = 0,
        )
        pred = pred[0] # type: ignore
        
        for prop in prop_configs:
            ase_prop_name = prop.ase_calculator_property_name()
            assert ase_prop_name is not None, (
                f"Property '{prop.name}' does not have an ASE calculator property name. "
                "This should have been checked when creating the MatterTuneCalculator. "
                "Please report this as a bug."
            )

            value = pred[prop.name].detach().cpu().numpy() # type: ignore
            value = value.astype(prop._numpy_dtype())
            value = prop.prepare_value_for_ase_calculator(value)

            self.results[ase_prop_name] = value
            
from tqdm import tqdm
import torch.distributed as dist


def quick_efs_evaluation(
    model: FinetuneModuleBase,
    atoms_list: list[Atoms],
    include_forces: bool = True,
    include_stresses: bool = True,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    metrics: list[str] = ["mae", "rmse"],
):
    
    calc = model.ase_calculator(device=device)
    
    energies = []
    energies_per_atom = []
    if include_forces:
        forces = []
    if include_stresses:
        stresses = []
    pred_energies = []
    pred_energies_per_atom = []
    if include_forces:
        pred_forces = []
    if include_stresses:
        pred_stresses = []
    for atoms in tqdm(atoms_list):
        energies.append(atoms.get_potential_energy())
        energies_per_atom.append(atoms.get_potential_energy() / len(atoms))
        if include_forces:
            forces.extend(np.array(atoms.get_forces()).tolist())
        if include_stresses:
            stresses.append(np.array(atoms.get_stress(voigt=False)).tolist())
        atoms.set_calculator(calc)
        pred_energies.append(atoms.get_potential_energy())
        pred_energies_per_atom.append(atoms.get_potential_energy() / len(atoms))
        if include_forces:
            pred_forces.extend(np.array(atoms.get_forces()).tolist())
        if include_stresses:
            pred_stresses.append(np.array(atoms.get_stress(voigt=False)).tolist())
    
    results = {}
    for metric in metrics:
        results[metric] = {}
        match metric.lower():
            case "mae":
                e_mae = torch.nn.L1Loss()(torch.tensor(energies_per_atom), torch.tensor(pred_energies_per_atom))
                results[metric]["e_mae"] = e_mae.item()
                if include_forces:
                    f_mae = torch.nn.L1Loss()(torch.tensor(forces), torch.tensor(pred_forces))
                    results[metric]["f_mae"] = f_mae.item()
                if include_stresses:
                    s_mae = torch.nn.L1Loss()(torch.tensor(stresses), torch.tensor(pred_stresses))
                    results[metric]["s_mae"] = s_mae.item()
            case "rmse":
                e_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(energies_per_atom), torch.tensor(pred_energies_per_atom)))
                results[metric]["e_rmse"] = e_rmse.item()
                if include_forces:
                    f_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(forces), torch.tensor(pred_forces)))
                    results[metric]["f_rmse"] = f_rmse.item()
                if include_stresses:
                    s_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(stresses), torch.tensor(pred_stresses)))
                    results[metric]["s_rmse"] = s_rmse.item()
            case "mse":
                e_mse = torch.nn.MSELoss()(torch.tensor(energies_per_atom), torch.tensor(pred_energies_per_atom))
                results[metric]["e_mse"] = e_mse.item()
                if include_forces:
                    f_mse = torch.nn.MSELoss()(torch.tensor(forces), torch.tensor(pred_forces))
                    results[metric]["f_mse"] = f_mse.item()
                if include_stresses:
                    s_mse = torch.nn.MSELoss()(torch.tensor(stresses), torch.tensor(pred_stresses))
                    results[metric]["s_mse"] = s_mse.item()
            case _:
                Warning(f"Metric '{metric}' not recognized. Skipping.")

    outputs = {
        "energies": energies,
        "energies_per_atom": energies_per_atom,
        "pred_energies": pred_energies,
        "pred_energies_per_atom": pred_energies_per_atom,
    }
    
    return results, outputs