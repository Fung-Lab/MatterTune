# Recipes

Recipes are modular components that modify the fine-tuning process in MatterTune. They provide a standardized way to implement advanced training techniques, particularly parameter-efficient fine-tuning methods, through Lightning callbacks.

## Overview

Recipes are configurable components that modify how models are trained in MatterTune. Each recipe provides a specific capability - like making training more memory-efficient, adding regularization, or enabling advanced optimization techniques.

Using a recipe is as simple as adding its configuration to your training setup:

```python
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(...),
    data=mt.configs.AutoSplitDataModuleConfig(...),
    recipes=[
        # List of recipe configurations
        mt.configs.MyRecipeConfig(...),
        mt.configs.AnotherRecipeConfig(...)
    ]
)
```

When training starts, each recipe is applied in order to modify the model, optimizer, or training loop. Recipes can be combined to create custom training pipelines that suit your specific needs.

## Available Recipes

### LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning technique that adds trainable rank decomposition matrices to model weights while keeping the original weights frozen.

API Reference: {py:class}`mattertune.configs.LoRARecipeConfig`

```python
import mattertune as mt

config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(...),
    data=mt.configs.AutoSplitDataModuleConfig(...),
    recipes=[
        mt.configs.LoRARecipeConfig(
            lora=mt.configs.LoraConfig(
                r=8,  # LoRA rank
                target_modules=["linear1", "linear2"],  # Layers to apply LoRA to
                lora_alpha=8,  # LoRA scaling factor
                lora_dropout=0.1  # Dropout probability
            )
        )
    ]
)
```

## Creating Custom Recipes

A recipe consists of two main components:

1. A configuration class that defines the parameters
2. A Lightning callback that integrates it into training

Here's how to create your own recipe:

1. Define a configuration class:
```python
class MyRecipeConfig(RecipeConfigBase):
    param1: int
    param2: float

    @classmethod
    @override
    def ensure_dependencies(cls):
        # Check for required packages
        if importlib.util.find_spec("some_package") is None:
            raise ImportError("Required package not found")
```

2. Implement the callback:
```python
class MyRecipeConfig(RecipeConfigBase):
    # ... Configuration class

    def create_lightning_callback(self):
        from lightning.pytorch.callbacks import LambdaCallback

        return LambdaCallback(
            on_train_start=lambda trainer, pl_module: print("Training started" + self.param1),
            on_train_end=lambda trainer, pl_module: print("Training ended" + self.param2)
        )
```

## Best Practices

1. **Configuration Validation**: Validate recipe parameters in `__post_init__`
2. **Dependency Management**: Use `ensure_dependencies` to check for required packages
3. **Error Handling**: Provide clear error messages for configuration issues
4. **Documentation**: Include docstrings explaining parameters and their effects
5. **Type Safety**: Use type hints for all parameters and return values

## Integration with Training

Recipes are automatically applied when training starts:

```python
tuner = mt.MatterTuner(config)
model, trainer = tuner.tune()  # Recipes are applied here
```

## Advanced Usage

Recipes can be combined and will be applied in order:

```python
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(...),
    data=mt.configs.AutoSplitDataModuleConfig(...),
    recipes=[
        mt.configs.LoRARecipeConfig(...),
        mt.configs.MyRecipeConfig(...),
    ]
)
```
