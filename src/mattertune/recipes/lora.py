from __future__ import annotations

import importlib.util
import logging
from typing import Any, Literal

import nshconfig as C
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from typing_extensions import final, override

from ..util import optional_import_error_message
from .base import RecipeConfigBase, recipe_registry

log = logging.getLogger(__name__)


class PeftConfig(C.Config):
    peft_type: str | None = None
    """Type of PEFT method being used."""

    task_type: str | None = None
    """Type of task being performed."""

    inference_mode: bool = False
    """Whether to use inference mode."""


class LoraConfig(PeftConfig):
    r: int = 8
    """LoRA attention dimension (rank)."""

    target_modules: list[str] | str | None = None
    """Names of modules to apply LoRA to. Can be a list of module names, a regex pattern, or 'all-linear'."""

    lora_alpha: int = 8
    """Alpha parameter for LoRA scaling."""

    lora_dropout: float = 0.0
    """Dropout probability for LoRA layers."""

    fan_in_fan_out: bool = False
    """Set True if target layer stores weights as (fan_in, fan_out)."""

    bias: Literal["none", "all", "lora_only"] = "none"
    """Bias type for LoRA. Controls which biases are updated during training."""

    use_rslora: bool = False
    """Whether to use Rank-Stabilized LoRA which sets adapter scaling to lora_alpha/sqrt(r)."""

    modules_to_save: list[str] | None = None
    """Additional modules to be trained and saved besides LoRA layers."""

    init_lora_weights: bool | Literal["gaussian"] = True
    """Initialization method for LoRA weights."""

    layers_to_transform: list[int] | int | None = None
    """Specific layer indices to apply LoRA transformation to."""

    layers_pattern: list[str] | str | None = None
    """Layer pattern name used with layers_to_transform."""

    rank_pattern: dict[str, Any] = {}
    """Mapping of layer names/patterns to custom ranks different from default r."""

    alpha_pattern: dict[str, Any] = {}
    """Mapping of layer names/patterns to custom alphas different from default lora_alpha."""

    def __post_init__(self):
        self.peft_type = "LORA"

        # Convert target_modules to set if it's a list
        self.target_modules = (
            list(set(self.target_modules))
            if isinstance(self.target_modules, list)
            else self.target_modules
        )

        # Validate target_modules and layers configurations
        if isinstance(self.target_modules, str):
            if self.layers_to_transform is not None:
                raise ValueError(
                    "layers_to_transform cannot be used when target_modules is a str"
                )
            if self.layers_pattern is not None:
                raise ValueError(
                    "layers_pattern cannot be used when target_modules is a str"
                )

    def _to_peft_config(self):
        """Convert this configuration to a PEFT LoraConfig instance."""
        with optional_import_error_message("peft"):
            from peft.tuners.lora import LoraConfig as PeftLoraConfig  # type: ignore[reportMissingImports] # noqa

        # Convert back to list if target_modules is a set
        return PeftLoraConfig(
            r=self.r,
            target_modules=self.target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            fan_in_fan_out=self.fan_in_fan_out,
            bias=self.bias,
            use_rslora=self.use_rslora,
            modules_to_save=self.modules_to_save,
            init_lora_weights=self.init_lora_weights,
            layers_to_transform=self.layers_to_transform,
            layers_pattern=self.layers_pattern,
            rank_pattern=self.rank_pattern,
            alpha_pattern=self.alpha_pattern,
            inference_mode=self.inference_mode,
            task_type=self.task_type,
        )


@recipe_registry.register
class LoRARecipeConfig(RecipeConfigBase):
    """
    Recipe for applying Low-Rank Adaptation (LoRA) to a model. LoRA is a method for
    fine-tuning pre-trained models via the injection of low-rank "adapter" weights
    into the model's linear layers. This allows for efficient fine-tuning of
    large models on small datasets, while preserving the pre-trained weights in the backbone.

    Reference: https://arxiv.org/abs/2106.09685
    """

    name: Literal["lora"] = "lora"
    """Discriminator for the LoRA recipe."""

    lora: LoraConfig
    """LoRA configuration."""

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the "peft" package is installed
        if importlib.util.find_spec("peft") is None:
            raise ImportError(
                "LoRARecipe requires the 'peft' package. To install it, run 'pip install peft'."
            )

    @override
    def create_lightning_callback(self):
        return LoRACallback(self)


@final
class LoRACallback(Callback):
    @override
    def __init__(self, config: LoRARecipeConfig):
        super().__init__()

        self.config = config

    @override
    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        from ..finetune.base import FinetuneModuleBase

        assert isinstance(
            pl_module, FinetuneModuleBase
        ), f"LoRARecipe requires a FinetuneModuleBase, got {type(pl_module)}="

        with optional_import_error_message("peft"):
            import peft  # type: ignore[reportMissingImports] # noqa

        # Convert the configuration to a PEFT LoraConfig instance
        lora = self.config.lora._to_peft_config()

        # Apply LoRA to the pre-trained backbone
        pl_module.apply_callable_to_backbone(
            lambda backbone: peft.inject_adapter_in_model(lora, backbone)
        )
        log.info("LoRA layers injected into the model")
