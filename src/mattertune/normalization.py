from __future__ import annotations

import argparse
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast, runtime_checkable

import ase
import nshconfig as C
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing_extensions import TypeAliasType, assert_never, override

from .finetune.properties import PropertyConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizationContext:
    """
    The normalization context contains all the information required to
        normalize and denormalize the properties. Currently, this only
        includes the compositions of the materials in the batch.

    This flexibility allows for the "Normalizer" interface to be used for
        other types of normalization, beyond just simple mean and standard
        deviation normalization. For example, subtracting linear references
        from total energies can be implemented using this interface.
    """

    compositions: torch.Tensor  # (num_samples, num_elements)
    """
    The compositions should be provided as an integer tensor of shape
        (batch_size, num_elements), where each row (i.e., `compositions[i]`)
        corresponds to the composition vector of the `i`-th material in the batch.

    The composition vector is a vector that maps each element to the number of
        atoms of that element in the material. For example, `compositions[:, 1]`
        corresponds to the number of Hydrogen atoms in each material in the batch,
        `compositions[:, 2]` corresponds to the number of Helium atoms, and so on.
    """


@runtime_checkable
class NormalizerModule(Protocol):
    def normalize(
        self,
        value: torch.Tensor,
        ctx: NormalizationContext,
    ) -> torch.Tensor:
        """Normalizes the input tensor using the normalizer's parameters and context.

        Args:
            value (torch.Tensor): The input tensor to be normalized
            ctx (NormalizationContext): Context containing compositions information

        Returns:
            torch.Tensor: The normalized tensor
        """
        ...

    def denormalize(
        self,
        value: torch.Tensor,
        ctx: NormalizationContext,
    ) -> torch.Tensor:
        """Denormalizes the input tensor using the normalizer's parameters and context.

        Args:
            value (torch.Tensor): The normalized tensor to be denormalized
            ctx (NormalizationContext): Context containing compositions information

        Returns:
            torch.Tensor: The denormalized tensor
        """
        ...


class NormalizerConfigBase(C.Config, ABC):
    @abstractmethod
    def create_normalizer_module(self) -> NormalizerModule: ...


class MeanStdNormalizerConfig(NormalizerConfigBase):
    mean: float
    """The mean of the property values."""

    std: float
    """The standard deviation of the property values."""


class MeanStdNormalizerModule(nn.Module, NormalizerModule, ABC):
    mean: torch.Tensor
    std: torch.Tensor

    @override
    def __init__(self, config: MeanStdNormalizerConfig):
        super().__init__()

        self.register_buffer("mean", torch.tensor(config.mean))
        self.register_buffer("std", torch.tensor(config.std))

    @override
    def normalize(self, value, ctx):
        return (value - self.mean) / self.std

    @override
    def denormalize(self, value, ctx):
        return value * self.std + self.mean


class RMSNormalizerConfig(NormalizerConfigBase):
    rms: float
    """The root mean square of the property values."""


class RMSNormalizerModule(nn.Module, NormalizerModule, ABC):
    rms: torch.Tensor

    @override
    def __init__(self, config: RMSNormalizerConfig):
        super().__init__()

        self.register_buffer("rms", torch.tensor(config.rms))

    @override
    def normalize(self, value, ctx):
        return value / self.rms

    @override
    def denormalize(self, value, ctx):
        return value * self.rms


class PerAtomReferencingNormalizerConfig(NormalizerConfigBase):
    per_atom_references: Mapping[int, float] | Sequence[float] | Path
    """The reference values for each element.

    - If a dictionary is provided, it maps atomic numbers to reference values
    - If a list is provided, it's a list of reference values indexed by atomic number
    - If a path is provided, it should point to a JSON file containing the references
    """

    def _references_as_dict(self) -> dict[int, float]:
        if isinstance(self.per_atom_references, Mapping):
            return dict(self.per_atom_references)
        elif isinstance(self.per_atom_references, Sequence):
            return {z: ref for z, ref in enumerate(self.per_atom_references)}
        else:
            with open(self.per_atom_references, "r") as f:
                return json.load(f)

    @override
    def create_normalizer_module(self) -> NormalizerModule:
        return PerAtomReferencingNormalizerModule(self)


class PerAtomReferencingNormalizerModule(nn.Module, NormalizerModule):
    references: torch.Tensor

    def __init__(self, config: PerAtomReferencingNormalizerConfig):
        super().__init__()

        references_dict = config._references_as_dict()
        max_atomic_number = max(references_dict.keys()) + 1
        references = torch.zeros(max_atomic_number)
        for z, ref in references_dict.items():
            references[z] = ref
        self.register_buffer("references", references)

    @override
    def normalize(self, value: torch.Tensor, ctx: NormalizationContext) -> torch.Tensor:
        # Compute references for each composition in the batch
        references = torch.einsum("ij,j->i", ctx.compositions, self.references)
        # Subtract references from values
        return value - references

    @override
    def denormalize(
        self, value: torch.Tensor, ctx: NormalizationContext
    ) -> torch.Tensor:
        # Add references back to get original values
        references = torch.einsum("ij,j->i", ctx.compositions, self.references)
        return value + references


class ComposeNormalizers(nn.Module, NormalizerModule):
    def __init__(self, normalizers: Sequence[NormalizerModule]):
        super().__init__()

        self.normalizers = nn.ModuleList(cast(list[nn.Module], normalizers))

    @override
    def normalize(self, value: torch.Tensor, ctx: NormalizationContext) -> torch.Tensor:
        for normalizer in self.normalizers:
            value = normalizer.normalize(value, ctx)
        return value

    @override
    def denormalize(
        self, value: torch.Tensor, ctx: NormalizationContext
    ) -> torch.Tensor:
        for normalizer in reversed(self.normalizers):
            value = normalizer.denormalize(value, ctx)
        return value


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
            assert_never(reference_model)

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
    from .main import MatterTunerConfig

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
    from .main import MatterTuneDataModule

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


NormalizerConfig = TypeAliasType(
    "NormalizerConfig",
    Annotated[
        MeanStdNormalizerConfig
        | RMSNormalizerConfig
        | PerAtomReferencingNormalizerConfig,
        C.Field(
            description="Configuration for normalizing and denormalizing a property."
        ),
    ],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-atom references for a property using a linear model."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="The path to the MatterTune config JSON file.",
    )
    parser.add_argument(
        "property",
        type=str,
        help="The name of the property for which to compute the per-atom references.",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="The path to save the computed per-atom references JSON file.",
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default="linear",
        choices=["linear", "ridge"],
        help="The type of reference model to use.",
    )
    parser.add_argument(
        "--reference-model-kwargs",
        type=json.loads,
        default={},
        help="The keyword arguments to pass to the reference model constructor.",
    )

    args = parser.parse_args()
    compute_per_atom_references_cli_main(args, parser)
