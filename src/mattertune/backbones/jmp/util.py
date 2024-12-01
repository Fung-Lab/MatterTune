from __future__ import annotations

import torch.nn as nn

from ...util import optional_import_error_message


def get_activation_cls(activation: str) -> type[nn.Module]:
    """
    Get the activation class from the activation name
    """
    match activation.lower():
        case "relu":
            return nn.ReLU
        case "silu" | "swish":
            return nn.SiLU
        case "scaled_silu" | "scaled_swish":
            with optional_import_error_message("jmp"):
                from jmp.models.gemnet.layers.base_layers import ScaledSiLU  # type: ignore[reportMissingImports] # noqa

            return ScaledSiLU
        case "tanh":
            return nn.Tanh
        case "sigmoid":
            return nn.Sigmoid
        case "identity":
            return nn.Identity
        case _:
            raise ValueError(f"Activation {activation} is not supported")
