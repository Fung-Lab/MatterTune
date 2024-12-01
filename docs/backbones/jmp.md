# JMP Backbone

The JMP backbone implements the Joint Multi-domain Pre-training (JMP) framework in MatterTune. This is a high-performance model architecture that combines message passing neural networks with transformers for accurate prediction of molecular and materials properties.

## Installation

Before using the JMP backbone, follow the installation instructions in the [jmp-backbone repository](https://github.com/nimashoghi/jmp-backbone/blob/lingyu-grad/README.md).

For development work with MatterTune, it's recommended to create a fresh conda environment:

```bash
conda create -n jmp-tune python=3.10
conda activate jmp-tune
```

## Key Features

- Hybrid architecture combining message passing networks with transformers
- Supports both molecular and periodic systems with flexible boundary conditions
- Highly optimized for both training and inference
- Support for property predictions:
  - Energy (extensive/intensive)
  - Forces (both conservative and non-conservative)
  - Stresses (both conservative and non-conservative)
  - Graph-level properties with customizable reduction

## Configuration

Here's a complete example showing how to configure the JMP backbone:

```python
from mattertune import configs as MC
from pathlib import Path

config = MC.MatterTunerConfig(
    model=MC.JMPBackboneConfig(
        # Required: Path to pre-trained checkpoint
        ckpt_path="path/to/jmp_checkpoint.pt",

        # Graph construction settings
        graph_computer=MC.JMPGraphComputerConfig(
            pbc=True,  # Set False for molecules
        ),

        # Properties to predict
        properties=[
            # Energy prediction
            MC.EnergyPropertyConfig(
                loss=MC.MAELossConfig(),
                loss_coefficient=1.0
            ),

            # Force prediction (conservative)
            MC.ForcesPropertyConfig(
                loss=MC.MAELossConfig(),
                loss_coefficient=10.0,
                conservative=True
            ),

            # Stress prediction (conservative)
            MC.StressesPropertyConfig(
                loss=MC.MAELossConfig(),
                loss_coefficient=1.0,
                conservative=True
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
    )
)
```

## Property Support

The JMP backbone supports the following property predictions:

### Energy Prediction
- Full support for extensive energy predictions
- Automated per-atom energy normalization
- Optional atomic reference energy subtraction

### Force Prediction
- Supports both conservative (energy-derived) and direct force prediction
- Configurable force scaling during training
- Automatic handling of periodic boundary conditions

### Stress Prediction
- Full support for stress tensor prediction
- Conservative (energy-derived) or direct stress computation
- Returns full 3x3 stress tensor with proper PBC handling

### Graph Properties
- Support for system-level property prediction
- Configurable "sum" or "mean" reduction over atomic features
- Suitable for both extensive and intensive properties

## Graph Construction Parameters

The JMP backbone uses a sophisticated multi-scale graph construction approach with several key parameters:

- `cutoffs`: Distance cutoffs for different interaction types
  - `main`: Primary interaction cutoff (typically 12.0 Å)
  - `aeaint`: Atomic energy interaction cutoff
  - `qint`: Charge interaction cutoff
  - `aint`: Auxiliary interaction cutoff

- `max_neighbors`: Maximum number of neighbors per interaction type
  - `main`: Primary interaction neighbors (typically 30)
  - `aeaint`: Atomic energy interaction neighbors
  - `qint`: Charge interaction neighbors
  - `aint`: Auxiliary interaction neighbors

## License

The JMP backbone is available under the CC BY-NC 4.0 License, which means:
- Free for academic and non-commercial use
- Required attribution when using or modifying the code
- Commercial use requires separate licensing

Please ensure compliance with the license terms before using this backbone in your projects.
