from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import nshconfig as C
import numpy as np
import torch
import torch.nn as nn
from typing_extensions import assert_never, final, override

if TYPE_CHECKING:
    from ...finetune.base import FinetuneModuleBase
    from ...finetune.properties import PropertyConfig
    from ..datamodule import MatterTuneDataModule


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


class ReferenceConfigBase(C.Config, ABC):
    @abstractmethod
    def create_referencer_module(
        self,
        base_module: FinetuneModuleBase,
        data_module: MatterTuneDataModule,
        property: PropertyConfig,
    ) -> ReferencerModule: ...


@final
class FixedReferenceConfig(ReferenceConfigBase):
    name: Literal["fixed_reference"] = "fixed_reference"

    references: dict[int, float]
    """The fixed reference values for each element."""

    @override
    def create_referencer_module(self, base_module, data_module, property):
        return FixedReferencerModule(self)


class FixedReferencerModule(ReferencerModuleBase):
    references: torch.Tensor

    @override
    def __init__(self, config: FixedReferenceConfig):
        self.config = config
        del config

        super().__init__(self.config.references)


@final
class OTFReferenceConfig(ReferenceConfigBase):
    name: Literal["otf_reference"] = "otf_reference"

    reference_model: Literal["linear", "ridge"]
    """The reference model to use."""

    reference_model_kwargs: dict[str, Any] = {}
    """The keyword arguments to pass to the reference model.
    These are directly passed to the constructor of the scikit-learn model."""

    @override
    def create_referencer_module(self, base_module, data_module, property):
        return OTFReferencerModule(self, base_module, data_module, property)


class OTFReferencerModule(ReferencerModuleBase):
    references: torch.Tensor

    @override
    def __init__(
        self,
        config: OTFReferenceConfig,
        base_module: FinetuneModuleBase,
        data_module: MatterTuneDataModule,
        property: PropertyConfig,
    ):
        self.config = config
        del config

        # To extract all labels (to fit the reference model), we need to
        #   iterate through the entire dataset in a batched
        #   manner.
        property_values: list[float] = []
        compositions: list[Counter[int]] = []

        # Make sure `prepare_data` and `setup` are called
        data_module.prepare_data()
        data_module.setup("fit")

        # After setup, `data_module.datasets` should be populated.
        # Get the train dataset from the data module
        if (dataset := data_module.datasets.get("train")) is None:
            raise ValueError("No training dataset found.")

        # Iterate through the dataset to extract all labels.
        for atoms in dataset:
            # Extract the composition from the `ase.Atoms` object
            composition = Counter(atoms.get_atomic_numbers())

            # The BaseModule interface only enforces label extraction
            #   for batched inputs. This is quite hacky, but it is what it is.
            data = base_module.atoms_to_data(atoms, has_labels=True)
            data = base_module.cpu_data_transform(data)
            batch = base_module.collate_fn([data])

            # Get the labels.
            labels = base_module.batch_to_labels(batch)
            label: torch.Tensor = labels[property.name]

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
        match self.config.reference_model:
            case "linear":
                from sklearn.linear_model import LinearRegression

                model = LinearRegression(
                    fit_intercept=False,
                    **self.config.reference_model_kwargs,
                )
            case "ridge":
                from sklearn.linear_model import Ridge

                model = Ridge(
                    fit_intercept=False,
                    **self.config.reference_model_kwargs,
                )
            case _:
                assert_never(self.config.reference_model)

        references = model.fit(compositions_matrix, torch.tensor(property_values)).coef_
        # references: (num_elements,)

        # Convert the reference to a dict[int, float]
        references_dict = {z: ref for z, ref in enumerate(references.tolist())}

        super().__init__(references_dict)
