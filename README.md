# MatterTune: A Unified Platform for Atomistic Foundation Model Fine-Tuning

[![Documentation Status](https://github.com/Fung-Lab/MatterTune/actions/workflows/docs.yml/badge.svg)](https://fung-lab.github.io/MatterTune/)

**[📚 Documentation](https://fung-lab.github.io/MatterTune/) | [🔧 Installation Guide](https://fung-lab.github.io/MatterTune/installation.html)**

MatterTune is a flexible and powerful machine learning library designed specifically for fine-tuning state-of-the-art chemistry models. It provides intuitive interfaces for computational chemists to fine-tune pre-trained models on their specific use cases.

## Features

- Pre-trained model support: JMP, EquiformerV2, M3GNet, ORB
- Multiple property predictions: energy, forces, stress, and custom properties
- Various dataset formats: XYZ, ASE databases, Materials Project, Matbench, and more
- Comprehensive training features with automated data splitting and logging

## Quick Start

```python
import mattertune as mt
from pathlib import Path

# Define the configuration for model, data, and training
config = mt.configs.MatterTunerConfig(
    # Configure the model: using JMP backbone with energy prediction
    model=mt.configs.JMPBackboneConfig(
        ckpt_path=Path("YOUR_CHECKPOINT_PATH"),  # Path to pre-trained model
        properties=[
            mt.configs.EnergyPropertyConfig(  # Configure energy prediction
                loss=mt.configs.MAELossConfig(),  # Using MAE loss
                loss_coefficient=1.0  # Weight for this property's loss
            )
        ],
    ),
    # Configure the data: loading from XYZ file with automatic train/val split
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=mt.configs.XYZDatasetConfig(
            src=Path("YOUR_XYZFILE_PATH")  # Path to your XYZ data
        ),
        train_split=0.8,  # Use 80% of data for training
        batch_size=32  # Process 32 structures per batch
    ),
    # Configure the training process
    trainer=mt.configs.TrainerConfig(
        max_epochs=10,  # Train for 10 epochs
        accelerator="gpu",  # Use GPU for training
        devices=[0]  # Use first GPU
    ),
)

# Create tuner and start training
tuner = mt.MatterTune(config)
model, trainer = tuner.tune()

# Save the fine-tuned model
trainer.save_checkpoint("finetuned_model.ckpt")
```

## License

MatterTune's core framework is licensed under the MIT License. Note that each supported model backbone is subject to its own licensing terms - see our documentation for details.

## Citation

Coming soon.
