# M3GNet Backbone

The M3GNet backbone implements the M3GNet model architecture in MatterTune. It provides a powerful graph neural network designed specifically for materials science applications.

## Overview

M3GNet supports predicting:
- Total energy (with energy conservation)
- Atomic forces (derived from energy)
- Stress tensors (derived from energy)

The model uses a three-body graph neural network architecture that captures both two-body and three-body interactions between atoms.

## Configuration

Configure the M3GNet backbone using:

```python
model = mt.configs.M3GNetBackboneConfig(
    # Path to pretrained checkpoint
    ckpt_path="path/to/checkpoint",

    # Graph computer settings
    graph_computer=mt.configs.M3GNetGraphComputerConfig(
        # Cutoff distance for neighbor list. If None, the cutoff is loaded from the checkpoint.
        cutoff=6.0,

        # Cutoff for three-body interactions. If None, the cutoff is loaded from the checkpoint.
        threebody_cutoff=4.0,

        # Whether to precompute line graphs
        pre_compute_line_graph=False
    ),

    # Properties to predict
    properties=[
        mt.configs.EnergyPropertyConfig(
            loss=mt.configs.MAELossConfig(),
            loss_coefficient=1.0
        ),
        mt.configs.ForcesPropertyConfig(
            loss=mt.configs.MAELossConfig(),
            loss_coefficient=0.1,
            conservative=True  # Forces derived from energy
        )
    ],

    # Training settings
    optimizer=mt.configs.AdamConfig(lr=1e-4)
)
```

### Key Parameters

- `ckpt_path`: Path to pretrained model checkpoint
- `graph_computer`: Controls graph construction:
  - `element_types`: Elements to include (defaults to all)
  - `cutoff`: Distance cutoff for neighbor list
  - `threebody_cutoff`: Cutoff for three-body interactions
  - `pre_compute_line_graph`: Whether to precompute line graphs
- `properties`: List of properties to predict
- `optimizer`: Optimizer configuration

## Implementation Details

The backbone is implemented in `M3GNetBackboneModule` which:

1. Loads the pretrained model using MatGL
2. Constructs atomic graphs with both two-body and three-body interactions
3. Handles property prediction with energy conservation
4. Manages normalization of inputs/outputs

Key features:
- Energy-conserving force prediction
- Three-body interactions for improved accuracy
- Efficient graph construction
- Support for periodic boundary conditions
