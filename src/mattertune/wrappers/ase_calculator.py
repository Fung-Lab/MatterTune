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
        batch = batch.to(self.model.device)
        
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