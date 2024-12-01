from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from typing_extensions import override

if TYPE_CHECKING:
    from ..finetune.properties import PropertyConfig
    from .property_predictor import MatterTunePropertyPredictor


class MatterTuneCalculator(Calculator):
    @override
    def __init__(self, property_predictor: MatterTunePropertyPredictor):
        super().__init__()

        self.property_predictor = property_predictor

        self.implemented_properties: list[str] = []
        self._ase_prop_to_config: dict[str, PropertyConfig] = {}

        for prop in self.property_predictor.lightning_module.hparams.properties:
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
        """Calculate properties for the given atoms object using the MatterTune property predictor.
        This method implements the calculation of properties like energy, forces, etc. for an ASE Atoms
        object using the underlying MatterTune property predictor. It converts between ASE and MatterTune
        property names and handles proper type conversions of the predicted values.

        Args:
            atoms (Atoms | None, optional): ASE Atoms object to calculate properties for. If None,
                uses previously set atoms. Defaults to None.
            properties (list[str] | None, optional): List of properties to calculate. If None,
                calculates all implemented properties. Defaults to None.
            system_changes (list[str] | None, optional): List of changes made to the system
                since last calculation. Used by ASE for caching. Defaults to None.
        Notes:
            - The method first ensures atoms and property names are properly set
            - Makes predictions using the MatterTune property predictor
            - Converts predictions from PyTorch tensors to appropriate numpy types
            - Maps MatterTune property names to ASE calculator property names
            - Stores results in the calculator's results dictionary
        Raises:
            AssertionError: If atoms is not set properly or if predictions are not in expected format
        """

        # Default properties to calculate.
        if properties is None:
            properties = copy.deepcopy(self.implemented_properties)

        # Call the parent class to set `self.atoms`.
        super().calculate(atoms, properties, system_changes)

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

        # Get the predictions.
        prop_configs = [self._ase_prop_to_config[prop] for prop in properties]
        predictions = self.property_predictor.predict(
            [self.atoms],
            prop_configs,
        )
        # The output of the predictor should be a list of predictions, where
        #   each prediction is a dictionary of properties. The PropertyPredictor class
        #   supports batch predictions, but we're only passing a single Atoms
        #   object here. So we expect a single prediction.
        assert len(predictions) == 1, "Expected a single prediction."
        [prediction] = predictions

        # Further, the output of the predictor is be a dictionary with the
        #   property names as keys. These property are should be the
        #   MatterTune property names (i.e., the names from the
        #   `lightning_module.hparams.properties[*].name` attribute), not the ASE
        #   calculator property names. Before feeding the properties to the
        #   ASE calculator, we need to convert the property names to the ASE
        #   calculator property names.
        for prop in prop_configs:
            ase_prop_name = prop.ase_calculator_property_name()
            assert ase_prop_name is not None, (
                f"Property '{prop.name}' does not have an ASE calculator property name. "
                "This should have been checked when creating the MatterTuneCalculator. "
                "Please report this as a bug."
            )

            # Update the ASE calculator results.
            # We also need to convert the prediction to the correct type.
            #   `PropertyPredictor.predict` returns the predictions as a
            #   `dict[str, torch.Tensor]`, but the ASE calculator expects the
            #   properties as numpy arrays/floats.
            value = prediction[prop.name].detach().to(torch.float32).cpu().numpy()
            value = value.astype(prop._numpy_dtype())

            # Finally, some properties may define their own conversion functions
            #   to do some final processing before setting the property value.
            #   For example, `energy` ends up being a scalar, so we call
            #   `value.item()` to get the scalar value. We handle this here.
            value = prop.prepare_value_for_ase_calculator(value)

            # Set the property value in the ASE calculator.
            self.results[ase_prop_name] = value
