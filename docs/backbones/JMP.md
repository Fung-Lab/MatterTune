# JMP Backbone

The JMP backbone is a deep learning model implementation based on the Joint Multi-domain Pre-training framework. It adapts the JMP model architecture for use within MatterTune.

## Overview

The JMP backbone supports predicting various materials properties including:

- Total energy (both regular and conservative)
- Atomic forces (both regular and conservative)
- Stress tensor (both regular and conservative)
- Graph-level scalar properties

## Configuration

The JMP backbone is configured through several key settings:

```yaml
model:
  name: jmp

  # Checkpoint path for loading pre-trained weights
  ckpt_path: path/to/checkpoint.pt

  # Graph construction settings
  graph_computer:
    pbc: true # Whether to use periodic boundary conditions
    cutoffs:
      main: 12.0
      aeaint: 12.0
      qint: 12.0
      aint: 12.0
    max_neighbors:
      main: 30
      aeaint: 20
      qint: 8
      aint: 1000
    per_graph_radius_graph: false
```

### Key Parameters

- `name`: Must be "jmp" to use this backbone
- `ckpt_path`: Path to pre-trained model checkpoint
- `graph_computer`: Settings for constructing the atomic graphs:
  - `pbc`: Whether to use periodic boundary conditions
  - `cutoffs`: Interaction cutoff distances for different components
  - `max_neighbors`: Maximum number of neighbors for different interaction types
  - `per_graph_radius_graph`: Whether to compute radius graph per system

## Usage Example

Here's a basic example of configuring the JMP backbone:

```python
from mattertune import MatterTuner

config = {
    "model": {
        "name": "jmp",
        "ckpt_path": "path/to/pretrained.pt",
        "graph_computer": {
            "pbc": True,
            "cutoffs": {"main": 12.0, "aeaint": 12.0, "qint": 12.0, "aint": 12.0},
            "max_neighbors": {
                "main": 30,
                "aeaint": 20,
                "qint": 8,
                "aint": 1000
            }
        }
    },
    # ... rest of config
}

tuner = MatterTuner(config)
```

## Property Prediction

The JMP backbone supports predicting:

1. System-level properties:
   - Total energy
   - Stress tensor
   - Other graph-level scalar properties

2. Atom-level properties:
   - Per-atom forces

Both conservative and non-conservative forces/stresses can be predicted.

## Implementation Details

The backbone is implemented in the `JMPBackboneModule` class which:

- Loads pre-trained weights from checkpoints
- Constructs atomic graphs using the configured settings
- Handles normalization of inputs/outputs
- Manages training and inference

The implementation includes:

- Graph construction
- Property prediction heads
- Normalization utilities
- ASE calculator interface

## Requirements

The JMP backbone requires:
- JMP library
- PyTorch Geometric
- PyTorch

These dependencies will be checked and validated when using the backbone.

## References

- JMP Paper/Documentation
- Source code in `mattertune/backbones/jmp/`
