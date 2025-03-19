# MatterTune: A Unified Platform for Atomistic Foundation Model Fine-Tuning

[![Documentation Status](https://github.com/Fung-Lab/MatterTune/actions/workflows/docs.yml/badge.svg)](https://fung-lab.github.io/MatterTune/)

**[📚 Documentation](https://fung-lab.github.io/MatterTune/) | [🔧 Installation Guide](https://fung-lab.github.io/MatterTune/installation.html)**

MatterTune is a flexible and powerful machine learning library designed specifically for fine-tuning state-of-the-art chemistry foundation models. It provides intuitive interfaces for computational chemists and materials scientists to fine-tune pre-trained models on their specific use cases.

## Features

- Pre-trained model support: JMP, EquiformerV2, MatterSim, ORB, and more to be added.
- Multiple property predictions: energy, forces, stress, and custom properties.
- Various supported dataset formats: XYZ, ASE databases, Materials Project, Matbench, and more.
- Comprehensive training features with automated data splitting and logging.

## Quick Start

```python
import mattertune as mt
import mattertune.configs as MC

# Phase 1: Fine-tuning the model
# -----------------------------

# Define the configuration for model, data, and training
config = mt.configs.MatterTunerConfig(
    # Configure the model: using JMP backbone with energy prediction
    model=MC.JMPBackboneConfig(
        pretrained_model = "jmp-s",  # Select pretrained model type
        graph_computer = MC.JMPGraphComputerConfig(pbc=True)
        properties = [
            MC.EnergyPropertyConfig(  # Configure energy prediction
                loss=MC.MAELossConfig(),  # Using MAE loss
                loss_coefficient=1.0  # Weight for this property's loss
            ),
            MC.ForcesPropertyConfig(
                loss=MC.MSELossConfig(), 
                conservative=False, 
                loss_coefficient=1.0
            )
        ],
        optimizer = MC.AdamWConfig(lr=8.0e-5)
    ),
    # Configure the data: loading from XYZ file with automatic train/val split
    data=MC.AutoSplitDataModuleConfig(
        dataset=MC.XYZDatasetConfig(
            src=Path("YOUR_XYZFILE_PATH")  # Path to your XYZ data
        ),
        train_split=0.8,  # Use 80% of data for training
        batch_size=32  # Process 32 structures per batch
    ),
    # Configure the training process
    trainer=MC.TrainerConfig(
        max_epochs=10,  # Train for 10 epochs
        accelerator="gpu",  # Use GPU for training
        devices=[0]  # Use first GPU
        additional_trainer_kwargs={
            "inference_mode": False,
        }
    ),
)

# Create tuner and start training
tuner = mt.MatterTune(config)
model, trainer = tuner.tune()

# Save the fine-tuned model
trainer.save_checkpoint("finetuned_model.ckpt")

# Phase 2: Using the fine-tuned model
# ----------------------------------

from ase.optimize import BFGS
from ase import Atoms

# Load the fine-tuned model
model = mt.backbones.JMPBackboneModule.load_from_checkpoint("finetuned_model.ckpt")

# Create an ASE calculator from the model
calculator = model.ase_calculator()

# Set up an atomic structure
atoms = Atoms('H2O',
             positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
             cell=[10, 10, 10],
             pbc=True)
atoms.calc = calculator

# Run geometry optimization
opt = BFGS(atoms)
opt.run(fmax=0.01)

# Get results
print("Final energy:", atoms.get_potential_energy())
print("Final forces:", atoms.get_forces())
```

## License

MatterTune's core framework is licensed under the MIT License. Note that each supported model backbone is subject to its own licensing terms - see our [license information page of the documentation](https://fung-lab.github.io/MatterTune/license.html) for more details.

## Citation

Coming soon.
