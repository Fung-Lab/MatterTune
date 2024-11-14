from __future__ import annotations

import argparse
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, runtime_checkable

import ase
import nshconfig as C
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing_extensions import TypeAliasType, assert_never, final, override

if TYPE_CHECKING:
    from ...finetune.properties import PropertyConfig

log = logging.getLogger(__name__)


@runtime_checkable
class ReferencerModule(Protocol):
    def compute_references(self, compositions: torch.Tensor) -> torch.Tensor:
        """
        Compute the reference energies for the given compositions.

        Args:
            compositions: The input compositions, with shape (num_samples, num_elements).

        Returns:
            The reference energies, with shape (num_samples,).
        """
        ...


class ReferencerModuleBase(nn.Module, ReferencerModule, ABC):
    references: torch.Tensor

    @override
    def __init__(self, config_references: dict[int, float]):
        super().__init__()

        max_atomic_number = max(config_references.keys()) + 1
        references = torch.zeros(max_atomic_number)
        for z, ref in config_references.items():
            references[z] = ref
        self.register_buffer("references", references)

    @override
    def compute_references(self, compositions: torch.Tensor) -> torch.Tensor:
        # compositions: (num_samples, num_elements)
        return torch.einsum("ij,j->i", compositions, self.references)


class PropertyReferencerConfigBase(C.Config, ABC):
    @abstractmethod
    def create_referencer_module(self) -> ReferencerModule: ...


@final
class FixedPerAtomPropertyReferencerConfig(PropertyReferencerConfigBase):
    name: Literal["fixed_per_atom_referencer"] = "fixed_per_atom_referencer"

    references: Mapping[int, float] | Sequence[float] | Path
    """The fixed reference values for each element.

    - If a dictionary is provided, it should map atomic numbers to reference values.
    - If a list is provided, it should be a list of reference values, where the index
        corresponds to the atomic number.
    - If a path is provided, it should be a path to a JSON file containing the reference values.
    """

    def _references_as_dict(self) -> dict[int, float]:
        if isinstance(self.references, Mapping):
            return dict(self.references)
        elif isinstance(self.references, Sequence):
            return {z: ref for z, ref in enumerate(self.references)}
        else:
            with open(self.references, "r") as f:
                return json.load(f)

    @override
    def create_referencer_module(self):
        return FixedPerAtomPropertyReferencerModule(self)


class FixedPerAtomPropertyReferencerModule(ReferencerModuleBase):
    references: torch.Tensor

    @override
    def __init__(self, config: FixedPerAtomPropertyReferencerConfig):
        self.config = config
        del config

        super().__init__(self.config._references_as_dict())


def compute_per_atom_references(
    dataset: Dataset[ase.Atoms],
    property: PropertyConfig,
    reference_model: Literal["linear", "ridge"],
    reference_model_kwargs: dict[str, Any],
):
    property_values: list[float] = []
    compositions: list[Counter[int]] = []

    # Iterate through the dataset to extract all labels.
    for atoms in dataset:
        # Extract the composition from the `ase.Atoms` object
        composition = Counter(atoms.get_atomic_numbers())

        # Get the property value
        label = property._from_ase_atoms_to_torch(atoms)

        # Make sure label is a scalar and convert to float
        assert (
            label.numel() == 1
        ), f"Label for property {property.name} is not a scalar. Shape: {label.shape}"

        property_values.append(float(label.item()))
        compositions.append(composition)

    # Convert the compositions to a matrix
    num_samples = len(compositions)
    num_elements = max(max(c.keys()) for c in compositions) + 1
    compositions_matrix = np.zeros((num_samples, num_elements))
    for i, composition in enumerate(compositions):
        for z, count in composition.items():
            compositions_matrix[i, z] = count

    # Fit the linear model
    match reference_model:
        case "linear":
            from sklearn.linear_model import LinearRegression

            model = LinearRegression(fit_intercept=False, **reference_model_kwargs)
        case "ridge":
            from sklearn.linear_model import Ridge

            model = Ridge(fit_intercept=False, **reference_model_kwargs)
        case _:
            assert_never(self.config.reference_model)

    references = model.fit(compositions_matrix, torch.tensor(property_values)).coef_
    # references: (num_elements,)

    # Convert the reference to a dict[int, float]
    references_dict = {z: ref for z, ref in enumerate(references.tolist())}

    return references_dict


def compute_per_atom_references_cli_main(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
):
    # Extract the necessary arguments
    config_arg: Path = args.config
    property_name_arg: str = args.property
    dest_arg: Path = args.dest

    # Load the fine-tuning config
    from ...main import MatterTunerConfig

    with open(config_arg, "r") as f:
        config = MatterTunerConfig.model_validate_json(f.read())

    # Extract the property config from the model config
    if (
        property := next(
            p for p in config.model.properties if p.name == property_name_arg
        )
    ) is None:
        parser.error(f"Property {property_name_arg} not found in the model config.")

    # Load the dataset based on the config
    from ...main import MatterTuneDataModule

    data_module = MatterTuneDataModule(config.data)
    data_module.prepare_data()
    data_module.setup("fit")

    # Get the train dataset or throw
    if (dataset := data_module.datasets.get("train")) is None:
        parser.error("The data module does not have a train dataset.")

    # Compute the reference values
    references_dict = compute_per_atom_references(
        dataset,
        property,
        args.reference_model,
        args.reference_model_kwargs,
    )

    # Print the reference values
    log.info(f"Computed reference values:\n{references_dict}")

    # Save the reference values to a JSON file
    with open(dest_arg, "w") as f:
        json.dump(references_dict, f)


PropertyReferencerConfig = TypeAliasType(
    "PropertyReferencerConfig",
    Annotated[
        FixedPerAtomPropertyReferencerConfig,
        C.Field(
            discriminator="name",
            description="The configuration for a property referencer.",
        ),
    ],
)
