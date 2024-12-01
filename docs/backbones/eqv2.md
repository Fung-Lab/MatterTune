# EquiformerV2 Backbone

The EquiformerV2 backbone implements Meta AI's EquiformerV2 model architecture in MatterTune. This is a state-of-the-art equivariant transformer model for molecular and materials property prediction, offering excellent performance across a wide range of chemical systems.

## Installation

Before using the EquiformerV2 backbone, you need to set up the required dependencies in a fresh conda environment:

```bash
conda create -n eqv2-tune python=3.10
conda activate eqv2-tune

# Install fairchem core
pip install "git+https://github.com/FAIR-Chem/fairchem.git@omat24#subdirectory=packages/fairchem-core" --no-deps

# Install dependencies
pip install ase "e3nn>=0.5" hydra-core lmdb numba "numpy>=1.26,<2.0" orjson \
    "pymatgen>=2023.10.3" submitit tensorboard "torch>=2.4" wandb torch_geometric \
    h5py netcdf4 opt-einsum spglib
```

## Key Features

- E(3)-equivariant transformer architecture for robust geometric predictions
- Support for both molecular and periodic systems
- Highly optimized implementation for efficient training and inference
- Pre-trained models available from Meta AI's OMAT24 release
- Support for property predictions:
  - Energy (extensive/intensive)
  - Forces (non-conservative)
  - Stresses (non-conservative)
  - System-level graph properties (with sum/mean reduction)

## Configuration

Here's a complete example showing how to configure the EquiformerV2 backbone:

```python
from mattertune import configs as MC
from pathlib import Path

config = MC.MatterTunerConfig(
    model=MC.EqV2BackboneConfig(
        # Required: Path to pre-trained checkpoint
        checkpoint_path="path/to/eqv2_checkpoint.pt",

        # Configure graph construction
        atoms_to_graph=MC.FAIRChemAtomsToGraphSystemConfig(
            cutoff=12.0,  # Angstroms
            max_neighbors=50,
            pbc=True  # Set False for molecules
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

The EquiformerV2 backbone supports the following property predictions:

### Energy Prediction
- Uses `EquiformerV2EnergyHead` for extensive energy predictions
- Always uses "sum" reduction over atomic contributions

### Force Prediction
- Uses `EquiformerV2ForceHead` for direct force prediction
- Currently only supports non-conservative forces (energy-derived forces coming soon)

### Stress Prediction
- Uses `Rank2SymmetricTensorHead` for stress tensor prediction
- Currently only supports non-conservative stresses
- Returns full 3x3 stress tensor

### Graph Properties
- Uses `EquiformerV2EnergyHead` with configurable reduction
- Supports "sum" or "mean" reduction over atomic features
- Suitable for intensive properties like bandgap

## Limitations

- Conservative forces and stresses not yet supported (coming in future release)
- Graph construction parameters must be manually specified (automatic loading from checkpoint coming soon)
- Per-species reference energy normalization not yet implemented

## Using Pre-trained Models

The EquiformerV2 backbone supports loading pre-trained models from Meta AI's OMAT24 release. Here's how to use them:

```python
config = C.MatterTunerConfig(
    model=C.EqV2BackboneConfig(
        checkpoint_path=C.CachedPath(
            uri='hf://fairchem/OMAT24/eqV2_31M_mp.pt'
        ),
        # ... rest of config ...
    )
)
```

Please visit the [Meta AI OMAT24 model release page on Hugging Face](https://huggingface.co/fairchem/OMAT24) for more details.

## License

The EquiformerV2 backbone is subject to Meta's Research License. Please ensure compliance with the license terms when using this backbone, especially for commercial applications.
