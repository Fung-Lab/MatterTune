# M3GNet Backbone

The M3GNet backbone implements the M3GNet model architecture in MatterTune. It provides a powerful graph neural network designed specifically for materials science applications. In MatterTune, we chose the M3GNet model implemented by MatGL and pretrained on MPTraj dataset. 

## Installation

```bash
conda create -n matgl-tune python=3.10 -y
pip install matgl
pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Key Features

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
        cutoff=5.0,

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

## Examples & Notebooks

A notebook tutorial about how to fine-tune M3GNet model can be found in ```notebooks/m3gnet-waterthermo.ipynb```([link](https://github.com/Fung-Lab/MatterTune/blob/main/notebooks/m3gnet-waterthermo.ipynb)). 

For advanced usage regarding fine-tuning models and applying them to downstream tasks (MD simulation for example), please refer to ```water-thermodynamics```([link](https://github.com/Fung-Lab/MatterTune/tree/main/examples/water-thermodynamics))

## License

We used the M3GNet model implemented in MatGL package, which is available under BSD 3-Clause License, which means redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
