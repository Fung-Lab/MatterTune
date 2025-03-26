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
    
    num_atoms: torch.Tensor  # (num_samples,)

    compositions: torch.Tensor  # (num_samples, num_elements)
    """
    The compositions should be provided as an integer tensor of shape
    `(batch_size, num_elements)`, where each row (i.e., `compositions[i]`)
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
    only_for_target: bool
    """
    Whether the normalizer should only be applied to the target property or to both predictions and targets.
    """
    @abstractmethod
    def create_normalizer_module(self) -> NormalizerModule: ...
    
    
class PerAtomNormalizerConfig(NormalizerConfigBase):
    only_for_target: bool = False
    @override
    def create_normalizer_module(self) -> NormalizerModule:
        return PerAtomNormalizerModule(self)

class PerAtomNormalizerModule(nn.Module, NormalizerModule, ABC):
    
    def __init__(self, config: PerAtomNormalizerConfig):
        super().__init__()
        self.only_for_target = config.only_for_target
    
    @override
    def normalize(self, value: torch.Tensor, ctx: NormalizationContext) -> torch.Tensor:
        if len(value.shape) == 1:
            return value / ctx.num_atoms
        else:
            return value / ctx.num_atoms[:, None]

    @override
    def denormalize(self, value: torch.Tensor, ctx: NormalizationContext) -> torch.Tensor:
        if len(value.shape) == 1:
            return value * ctx.num_atoms
        else:
            return value * ctx.num_atoms[:, None]


class MeanStdNormalizerConfig(NormalizerConfigBase):
    only_for_target: bool = True
    
    mean: float
    """The mean of the property values."""

    std: float
    """The standard deviation of the property values."""

    @override
    def create_normalizer_module(self) -> MeanStdNormalizerModule:
        return MeanStdNormalizerModule(self)


class MeanStdNormalizerModule(nn.Module, NormalizerModule, ABC):
    mean: torch.Tensor
    std: torch.Tensor

    @override
    def __init__(self, config: MeanStdNormalizerConfig):
        super().__init__()

        self.register_buffer("mean", torch.tensor(config.mean))
        self.register_buffer("std", torch.tensor(config.std))
        self.only_for_target = config.only_for_target

    @override
    def normalize(self, value, ctx):
        return (value - self.mean) / self.std

    @override
    def denormalize(self, value, ctx):
        return value * self.std + self.mean


class RMSNormalizerConfig(NormalizerConfigBase):
    only_for_target: bool = True
    
    rms: float
    """The root mean square of the property values."""

    @override
    def create_normalizer_module(self) -> RMSNormalizerModule:
        return RMSNormalizerModule(self)


class RMSNormalizerModule(nn.Module, NormalizerModule, ABC):
    rms: torch.Tensor

    @override
    def __init__(self, config: RMSNormalizerConfig):
        super().__init__()

        self.register_buffer("rms", torch.tensor(config.rms))
        self.only_for_target = config.only_for_target

    @override
    def normalize(self, value, ctx):
        return value / self.rms

    @override
    def denormalize(self, value, ctx):
        return value * self.rms


class PerAtomReferencingNormalizerConfig(NormalizerConfigBase):
    only_for_target: bool = True
    
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
                per_atom_references = json.load(f)
                per_atom_references = {
                    int(k): v for k, v in per_atom_references.items()
                }
            return per_atom_references

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
        ## delete reference with key 0
        references = references[1:]
        self.register_buffer("references", references)
        self.only_for_target = config.only_for_target

    @override
    def normalize(self, value: torch.Tensor, ctx: NormalizationContext) -> torch.Tensor:
        # Compute references for each composition in the batch
        references = self.references
        max_atomic_number = len(references)
        compositions = ctx.compositions[:, :max_atomic_number].to(references.dtype)
        references = torch.einsum("ij,j->i", compositions, references).reshape(
            value.shape
        )
        # Subtract references from values
        return value - references

    @override
    def denormalize(
        self, value: torch.Tensor, ctx: NormalizationContext
    ) -> torch.Tensor:
        # Add references back to get original valuesreferences = self.references
        references = self.references
        max_atomic_number = len(references)
        compositions = ctx.compositions[:, :max_atomic_number].to(references.dtype)
        references = torch.einsum("ij,j->i", compositions, references).reshape(
            value.shape
        )
        return value + references


class ComposeNormalizers(nn.Module):
    def __init__(self, normalizers: Sequence[NormalizerModule]):
        super().__init__()

        self.normalizers = nn.ModuleList(cast(list[nn.Module], normalizers))

    def normalize(
        self, prediction: torch.Tensor, target: torch.Tensor, ctx: NormalizationContext) -> tuple[torch.Tensor, torch.Tensor]:
        for normalizer in self.normalizers:
            if normalizer.only_for_target:
                ## When only_for target, model's prediction is already normalized
                target = normalizer.normalize(target, ctx)
            else:
                ## When not only_for_target, model's prediction is not normalized, where we need to normalize both prediction and target
                prediction = normalizer.normalize(prediction, ctx)
                target = normalizer.normalize(target, ctx)
        return prediction, target

    def denormalize(
        self, prediction: torch.Tensor, target: torch.Tensor, ctx: NormalizationContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for normalizer in reversed(self.normalizers):
            if normalizer.only_for_target:
                ## When only_for target, model's prediction is already normalized
                ## So we need to denormalize the target
                target = normalizer.denormalize(target, ctx)
            else:
                ## When not only_for_target, model's prediction is both normalized, where we need to denormalize both prediction and target
                prediction = normalizer.denormalize(prediction, ctx)
                target = normalizer.denormalize(target, ctx)
        return prediction, target

    def denormalize_predict(
        self, prediction: torch.Tensor, ctx: NormalizationContext
    ) -> torch.Tensor:
        ## NOTE: in denormalize, we should denormalize both prediction and target whether or not the normalizer is only for target
        ## This is because even if the normalizer is only for target, model's prediction is the normalized value, so we need to denormalize it
        for normalizer in reversed(self.normalizers):
            if normalizer.only_for_target:
                ## When only_for target, model's prediction is already normalized
                ## So we need to denormalize the prediction
                prediction = normalizer.denormalize(prediction, ctx)
            else:
                ## When not only_for_target, model's prediction is not normalized, just pass
                pass
        return prediction


def compute_per_atom_references(
    dataset: Dataset[ase.Atoms],
    property: PropertyConfig,
    reference_model: Literal["linear", "ridge"],
    reference_model_kwargs: dict[str, Any] = {},
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
    references_dict = {int(z): ref for z, ref in enumerate(references.tolist())}
    ## delete reference with key 0
    del references_dict[0]
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
        PerAtomNormalizerConfig
        | MeanStdNormalizerConfig
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
