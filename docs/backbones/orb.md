# ORB Backbone

The ORB backbone implements the Orbital Neural Networks model architecture in MatterTune. This is a state-of-the-art graph neural network designed specifically for molecular and materials property prediction, with excellent performance across diverse chemical systems.

## Installation

Before using the ORB backbone, you need to install the required dependencies:

```bash
pip install "orb_models@git+https://github.com/nimashoghi/orb-models.git"
```

## Key Features

- Advanced graph neural network architecture optimized for materials
- Support for both molecular and periodic systems
- Highly efficient implementation for fast training and inference
- Pre-trained models available from the orb-models package
- Support for property predictions:
  - Energy (extensive/intensive)
  - Forces (non-conservative)
  - Stresses (non-conservative)
  - System-level graph properties (with configurable reduction)

## Configuration

Here's a complete example showing how to configure the ORB backbone:

```python
from mattertune import configs as MC
from pathlib import Path

config = MC.MatterTunerConfig(
    model=MC.ORBBackboneConfig(
        # Required: Name of pre-trained model
        pretrained_model="orb-v2",

        # Configure graph construction
        system=MC.ORBSystemConfig(
            radius=10.0,  # Angstroms
            max_num_neighbors=20
        ),

        # Properties to predict
        properties=[
            # Energy prediction
            MC.EnergyPropertyConfig(
                loss=MC.MAELossConfig(),
                loss_coefficient=1.0
            ),

            # Force prediction (non-conservative)
            MC.ForcesPropertyConfig(
                loss=MC.MAELossConfig(),
                loss_coefficient=10.0,
                conservative=False
            ),

            # Stress prediction (non-conservative)
            MC.StressesPropertyConfig(
                loss=MC.MAELossConfig(),
                loss_coefficient=1.0,
                conservative=False
            ),

            # System-level property prediction
            MC.GraphPropertyConfig(
                name="bandgap",
                loss=MC.MAELossConfig(),
                loss_coefficient=1.0,
                reduction="mean"  # or "sum"
            )
        ],

        # Optimizer settings
        optimizer=MC.AdamWConfig(lr=1e-4),

        # Optional: Learning rate scheduler
        lr_scheduler=MC.CosineAnnealingLRConfig(
            T_max=100,
            eta_min=1e-6
        )
    ),

    # ... data and trainer configs ...
)
```

## Property Support

The ORB backbone supports the following property predictions:

### Energy Prediction
- Uses `EnergyHead` for extensive energy predictions
- Supports automated per-atom energy normalization
- Optional atomic reference energy subtraction

### Force Prediction
- Uses `NodeHead` for direct force prediction
- Currently only supports non-conservative forces
- Configurable force scaling during training

### Stress Prediction
- Uses `GraphHead` for stress tensor prediction
- Currently only supports non-conservative stresses
- Returns full 3x3 stress tensor

### Graph Properties
- Uses `GraphHead` with configurable reduction
- Supports "sum" or "mean" reduction over atomic features
- Suitable for both extensive and intensive properties

## Graph Construction Parameters

The ORB backbone uses a sophisticated graph construction approach with two key parameters:

- `radius`: The cutoff distance for including neighbors in the graph (typically 10.0 Å)
- `max_num_neighbors`: Maximum number of neighbors per atom to include (typically 20)

## Limitations

- Conservative forces and stresses not supported
- Limited to fixed graph construction parameters
- No direct support for charge predictions
- Reference energy normalization requires manual configuration

## Using Pre-trained Models

The ORB backbone supports loading pre-trained models from the orb-models package. Available models include:

- `orb-v2`: General-purpose model trained on materials data
- `orb-qm9`: Model specialized for molecular systems
- `orb-mp`: Model specialized for crystalline materials

```python
config = MC.MatterTunerConfig(
    model=MC.ORBBackboneConfig(
        pretrained_model="orb-v2",
        # ... rest of config ...
    )
)
```

## License

The ORB backbone is available under the Apache 2.0 License, which allows both academic and commercial use with proper attribution.
