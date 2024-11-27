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
    }
}

tuner = MatterTuner(config)
