from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch.nn as nn
import torchmetrics
from typing_extensions import override

if TYPE_CHECKING:
    from .properties import PropertyConfig


class MetricBase(nn.Module, ABC):
    @override
    def __init__(self, property_name: str):
        super().__init__()

        self.property_name = property_name

    @abstractmethod
    def forward(
        self, prediction: dict[str, Any], ground_truth: dict[str, Any]
    ) -> Mapping[str, torchmetrics.Metric]: ...


class GraphPropertyMetrics(MetricBase):
    @override
    def __init__(self, property_name: str):
        super().__init__(property_name)

        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError(squared=True)
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.r2 = torchmetrics.R2Score()

    @override
    def forward(
        self,
        prediction: dict[str, Any],
        ground_truth: dict[str, Any],
    ):
        y_hat, y = prediction[self.property_name], ground_truth[self.property_name]

        self.mae(y_hat, y)
        self.mse(y_hat, y)
        self.rmse(y_hat, y)
        self.r2(y_hat, y)

        return {
            f"{self.property_name}_mae": self.mae,
            f"{self.property_name}_mse": self.mse,
            f"{self.property_name}_rmse": self.rmse,
            f"{self.property_name}_r2": self.r2,
        }


class FinetuneMetrics(nn.Module):
    def __init__(
        self,
        properties: Mapping[str, PropertyConfig],
        metric_prefix: str = "",
    ):
        super().__init__()

        self.metric_modules = nn.ModuleDict(
            {
                property_name: property_config.metric_cls()(property_name)
                for property_name, property_config in properties.items()
            }
        )

        self.metric_prefix = metric_prefix

    @override
    def forward(
        self, prediction: dict[str, Any], ground_truth: dict[str, Any]
    ) -> Mapping[str, torchmetrics.Metric]:
        metrics = {}

        for property_name, metric_module in self.metric_modules.items():
            metrics.update(metric_module(prediction, ground_truth))

        return {
            f"{self.metric_prefix}{metric_name}": metric
            for metric_name, metric in metrics.items()
        }
