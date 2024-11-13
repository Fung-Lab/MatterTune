from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated

import nshconfig as C
import numpy as np
import torch
from typing_extensions import TypeAliasType, TypeVar, override

TData = TypeVar("TData", int, float, np.ndarray, torch.Tensor, infer_variance=True)


class NormalizationPropertyConfigBase(C.Config, ABC):
    @abstractmethod
    def normalize(self, data: TData) -> TData: ...

    @abstractmethod
    def denormalize(self, data: TData) -> TData: ...


class MeanStdNormalizationPropertyConfig(NormalizationPropertyConfigBase):
    mean: float
    """The mean of the data."""

    std: float
    """The standard deviation of the data."""

    @override
    def normalize(self, data: TData):
        return (data - self.mean) / self.std

    @override
    def denormalize(self, data: TData):
        return data * self.std + self.mean


class RMSNormalizationPropertyConfig(NormalizationPropertyConfigBase):
    rms: float
    """The root mean square of the data."""

    @override
    def normalize(self, data: TData):
        return data / self.rms

    @override
    def denormalize(self, data: TData):
        return data * self.rms


NormalizationPropertyConfig = TypeAliasType(
    "NormalizationPropertyConfig",
    Annotated[
        NormalizationPropertyConfigBase,
        C.Field(
            description="Configuration for normalizing and denormalizing a property."
        ),
    ],
)

NormalizationConfig = TypeAliasType(
    "NormalizationConfig", dict[str, NormalizationPropertyConfig]
)
