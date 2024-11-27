# EquiformerV2 Backbone

The EquiformerV2 backbone implements the EquiformerV2 model architecture in MatterTune. This is a state-of-the-art equivariant transformer model for molecular and materials property prediction.

```{note}
This documentation is currently under development. Please check back later for complete documentation of the EquiformerV2 backbone.
```

## Key Features

- Equivariant transformer architecture
- Support for both molecular and periodic systems
- State-of-the-art performance on various property prediction tasks

## Basic Usage

```python
from mattertune import MatterTuner
from mattertune.configs import EquiformerV2BackboneConfig

config = {
    "model": {
        "name": "eqv2",
        "ckpt_path": "path/to/pretrained.pt",
        # Additional configuration options will be documented soon
    }
}

tuner = MatterTuner(config)
```
