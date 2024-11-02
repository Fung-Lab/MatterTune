from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ase import Atoms
from ase.calculators.calculator import Calculator
from typing_extensions import override

if TYPE_CHECKING:
    from ..finetune.properties import PropertyConfig
    from .potential import MatterTunePotential


class MatterTuneCalculator(Calculator):
    @override
    def __init__(self, potential: MatterTunePotential):
        super().__init__()

        self.potential = potential

        self.implemented_properties: list[str] = []
        self._ase_prop_to_config: dict[str, PropertyConfig] = {}

        for prop in self.potential.lightning_module.hparams.properties:
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
        predictions = self.potential.predict(
            [self.atoms],
            prop_configs,
        )
        # The output of the potential should be a list of predictions, where
        #   each prediction is a dictionary of properties. The Potential class
        #   supports batch predictions, but we're only passing a single Atoms
        #   object here. So we expect a single prediction.
        assert len(predictions) == 1, "Expected a single prediction."
        [prediction] = predictions

        # Further, the output of the potential is be a dictionary with the
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
            #   `Potential.predict` returns the predictions as a
            #   `dict[str, torch.Tensor]`, but the ASE calculator expects the
            #   properties as numpy arrays/floats.
            value = prediction[prop.name].detach().cpu().numpy()
            value = value.astype(prop._numpy_dtype())

            # Finally, some properties may define their own conversion functions
            #   to do some final processing before setting the property value.
            #   For example, `energy` ends up being a scalar, so we call
            #   `value.item()` to get the scalar value. We handle this here.
            value = prop.prepare_value_for_ase_calculator(value)

            # Set the property value in the ASE calculator.
            self.results[ase_prop_name] = value
