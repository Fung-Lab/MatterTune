# MatterTune: A Unified Platform for Atomistic Foundation Model Fine-Tuning

MatterTune is a flexible and powerful machine learning library designed specifically for fine-tuning state-of-the-art chemistry models. It provides intuitive interfaces for computational chemists to fine-tune pre-trained models on their specific use cases.

## Table of Contents
- [MatterTune: A Unified Platform for Atomistic Foundation Model Fine-Tuning](#mattertune-a-unified-platform-for-atomistic-foundation-model-fine-tuning)
    - [Table of Contents](#table-of-contents)
    - [Motivation](#motivation)
    - [Features](#features)
    - [Installation](#installation)
    - [Quick Start: Fine-Tuning a Pre-trained Model](#quick-start-fine-tuning-a-pre-trained-model)
        - [Training Process](#training-process)
        - [Checkpoints](#checkpoints)
    - [Model Usage](#model-usage)
        - [Making Predictions](#making-predictions)
        - [Using as ASE Calculator](#using-as-ase-calculator)
    - [Training Configuration](#training-configuration)
        - [Model Configuration](#model-configuration)
        - [Data Configuration](#data-configuration)
        - [Training Process Configuration](#training-process-configuration)
        - [Configuration Management](#configuration-management)
    - [Extending MatterTune](#extending-mattertune)
    - [Contributing](#contributing)
    - [License](#license)
    - [Citation](#citation)


## Motivation

Atomistic Foundation Models have emerged as powerful tools in molecular and materials science. However, the diverse implementations of these open-source models, with their varying architectures and interfaces, create significant barriers for customized fine-tuning and downstream applications.

MatterTune is a comprehensive platform that addresses these challenges through systematic yet general abstraction of Atomistic Foundation Model architectures. By adopting a modular design philosophy, MatterTune provides flexible and concise user interfaces that enable intuitive and efficient fine-tuning workflows. The platform features:

- Standardized abstractions of model architectures while maintaining generality
- Modular design for maximum flexibility and extensibility
- Streamlined user interfaces for fine-tuning procedures
- Integrated downstream task interfaces for:
    - Molecular dynamics simulations
    - Structure optimization
    - Property screening
    - And more...

Through these features, MatterTune significantly lowers the technical barriers for researchers and practitioners working with Atomistic Foundation Models, enabling them to focus on their scientific objectives rather than implementation details.


## Features

- **Pre-trained Model Support**: Seamlessly work with multiple state-of-the-art pre-trained models including:
  - JMP
  - EquiformerV2
  - M3GNet
  - ORB

- **Flexible Property Predictions**: Support for various molecular and materials properties:
  - Energy prediction
  - Force prediction (both conservative and non-conservative)
  - Stress tensor prediction
  - Custom system-level property predictions

- **Data Processing**: Built-in support for multiple data formats:
  - XYZ files
  - ASE databases
  - Materials Project database
  - Matbench datasets
  - Custom datasets

- **Training Features**:
  - Automated train/validation splitting
  - Multiple loss functions (MAE, MSE, Huber, L2-MAE)
  - Property normalization and scaling
  - Early stopping and model checkpointing
  - Comprehensive logging with WandB, TensorBoard, and CSV support

## Installation

```bash
pip install mattertune
```

Note: MatterTune requires PyTorch >= 2.0 and additional backbone-specific dependencies. When using specific backbones, you'll need:
- JMP backbone: `pip install jmp`
- EquiformerV2 backbone: `pip install fairchem`
- M3GNet backbone: `pip install matgl`
- ORB backbone: `pip install "orb-models@git+https://github.com/orbital-materials/orb-models.git"`

For development installation:

```bash
git clone https://github.com/nimashoghi/mattertune.git
cd mattertune
pip install -e .
```

## Quick Start: Fine-Tuning a Pre-trained Model

Here's a simple example of fine-tuning a pre-trained model for energy prediction:

```python
import mattertune as mt

# Define configuration
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(
        ckpt_path="path/to/pretrained/model.pt",
        properties=[
            mt.configs.EnergyPropertyConfig(
                loss=mt.configs.MAELossConfig(),
                loss_coefficient=1.0
            )
        ],
        optimizer=mt.configs.AdamWConfig(lr=1e-4)
    ),
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=mt.configs.XYZDatasetConfig(
            src="path/to/your/data.xyz"
        ),
        train_split=0.8,
        batch_size=32
    )
)

# Create tuner and train
tuner = mt.MatterTuner(config)
model, trainer = tuner.tune()

# Save the fine-tuned model
trainer.save_checkpoint("finetuned_model.ckpt")
```

### Training Process
MatterTune will:
- Automatically split your data into training/validation sets
- Log training metrics to your chosen logger (WandB/TensorBoard/CSV)
- Save model checkpoints periodically
- Stop training early if validation metrics stop improving

You can monitor training progress in real-time through your chosen logging interface.

### Checkpoints
MatterTune uses PyTorch Lightning for training and automatically saves checkpoints in the `lightning_logs` directory. The latest checkpoint can be found at:
```
lightning_logs/version_X/checkpoints/
```

Additionally, you can save the model manually using the `trainer.save_checkpoint` method.

## Model Usage

After training, you can use the model for predictions. To load a saved model checkpoint (from a previous fine-tuning run), you can use the `load_from_checkpoint` method:

```python
from mattertune.backbones import JMPBackboneModule

model = JMPBackboneModule.load_from_checkpoint("path/to/checkpoint.ckpt")
```

### Making Predictions
The `Potential` interface provides a simple way to make predictions for a single or batch of atoms:

```python
from ase import Atoms
import torch

# Create an ASE Atoms object
atoms1 = Atoms('H2O',
              positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
              cell=[10, 10, 10],
              pbc=True)

atoms2 = Atoms('H2O',
                positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                cell=[10, 10, 10],
                pbc=True)
# Get predictions using the model's potential interface
potential = model.potential()
predictions = potential.predict([atoms1, atoms2], ["energy", "forces"])

print("Energy:", predictions[0]["energy"], predictions[1]["energy"])
print("Forces:", predictions[0]["forces"], predictions[1]["forces"])
```

### Using as ASE Calculator
Our ASE calculator interface allows you to use the model for molecular dynamics or geometry optimization:

```python
from ase.optimize import BFGS

# Create calculator from model
calculator = model.ase_calculator()

# Set up atoms and calculator
atoms = Atoms('H2O',
              positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
              cell=[10, 10, 10],
              pbc=True)
atoms.calc = calculator

# Run geometry optimization
opt = BFGS(atoms)
opt.run(fmax=0.01)

# Get optimized results
print("Final energy:", atoms.get_potential_energy())
print("Final forces:", atoms.get_forces())
```

## Training Configuration

MatterTune uses a comprehensive configuration system to control all aspects of training. Here are the key components:

### Model Configuration
Control the model architecture and training parameters:
```python
model = mt.configs.JMPBackboneConfig(
    # Specify pre-trained model checkpoint
    ckpt_path="path/to/pretrained/model.pt",

    # Define properties to predict
    properties=[
        mt.configs.EnergyPropertyConfig(
            loss=mt.configs.MAELossConfig(),
            loss_coefficient=1.0
        ),
        mt.configs.ForcesPropertyConfig(
            loss=mt.configs.MAELossConfig(),
            loss_coefficient=10.0,
            conservative=True  # Use energy-conserving force prediction
        )
    ],

    # Configure optimizer
    optimizer=mt.configs.AdamWConfig(lr=1e-4),

    # Optional: Configure learning rate scheduler
    lr_scheduler=mt.configs.CosineAnnealingLRConfig(
        T_max=100,  # Number of epochs
        eta_min=1e-6  # Minimum learning rate
    )
)
```

### Data Configuration
Configure data loading and processing:
```python
data = mt.configs.AutoSplitDataModuleConfig(
    # Specify dataset source
    dataset=mt.configs.XYZDatasetConfig(
        src="path/to/your/data.xyz"
    ),

    # Control data splitting
    train_split=0.8,  # 80% for training

    # Configure batch size and loading
    batch_size=32,
    num_workers=4,  # Number of data loading workers
    pin_memory=True  # Optimize GPU transfer
)
```

### Training Process Configuration
Control the training loop behavior:
```python
trainer = mt.configs.TrainerConfig(
    # Hardware configuration
    accelerator="gpu",
    devices=[0, 1],  # Use GPUs 0 and 1

    # Training stopping criteria
    max_epochs=100,
    # OR: max_steps=1000,  # Stop after 1000 steps
    # OR: max_time=datetime.timedelta(hours=1),  # Stop after 1 hour

    # Validation frequency
    check_val_every_n_epoch=1,

    # Gradient clipping: Prevent exploding gradients
    gradient_clip_val=1.0,

    # Early stopping configuration: Stop if validation loss does not improve
    early_stopping=mt.configs.EarlyStoppingConfig(
        monitor="val/energy_mae",
        patience=20,
        mode="min"
    ),

    # Model checkpointing: Save best model based on validation loss
    checkpoint=mt.configs.ModelCheckpointConfig(
        monitor="val/energy_mae",
        save_top_k=1,
        mode="min"
    ),

    # Configure logging
    loggers=[
        mt.configs.WandbLoggerConfig(
            project="my-project",
            name="experiment-1"
        )
    ]
)

# Combine all configurations
config = mt.configs.MatterTunerConfig(
    model=model,
    data=data,
    trainer=trainer
)
```

These configurations can be customized based on your specific needs. The configuration system ensures that all parameters are properly validated and provides clear error messages if any settings are invalid.

When training begins, MatterTune will:
1. Set up the model and data loading based on your configuration
2. Initialize logging and checkpointing
3. Begin the training loop with the specified parameters
4. Save checkpoints and log metrics throughout training
5. Stop when either max_epochs is reached or early stopping criteria are met

You can monitor training progress through your configured logger (WandB/TensorBoard/CSV).

### Configuration Management

MatterTune uses [`nshconfig`](https://github.com/nimashoghi/nshconfig) (a type-safe configuration management library built on Pydantic) to handle all configuration. This provides several benefits:

- Full type checking of all hyperparameters
- Runtime validation of configuration values
- Clear error messages when configurations are invalid
- Multiple ways to create and load configurations

You can define configurations in three ways:

1. **Direct Construction** (as shown in examples above):
```python
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(...),
    data=mt.configs.AutoSplitDataModuleConfig(...),
    trainer=mt.configs.TrainerConfig(...)
)
```

2. **Loading from Files/Dictionaries**:
```python
# Load from YAML
config = mt.configs.MatterTunerConfig.from_yaml('/path/to/config.yaml')

# Load from JSON
config = mt.configs.MatterTunerConfig.from_json('/path/to/config.json')

# Load from dictionary
config = mt.configs.MatterTunerConfig.from_dict({
    'model': {...},
    'data': {...},
    'trainer': {...}
})
```

3. **Using Draft Configs** (a Pythonic way to create configurations step-by-step and finalize them at the end --- see the [`nshconfig`](https://github.com/nimashoghi/nshconfig) documentation for more details):
```python
# Create a draft config
config = mt.configs.MatterTunerConfig.draft()

# Set values progressively
config.model = mt.configs.JMPBackboneConfig.draft()
config.model.ckpt_path = "path/to/model.pt"
# ... set other values ...

# Finalize the config
final_config = config.finalize()
```

For more advanced configuration management features, including draft configs and dynamic type registration, see the [nshconfig documentation](https://github.com/nimashoghi/nshconfig).

## Extending MatterTune

MatterTune is designed to be extensible and customizable. You can create custom property configurations, datasets, and backbones by subclassing the provided base classes. Please visit the [advanced usage documentation](ADVANCED_USAGE.md) for more details.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

MatterTune's core framework is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Note that each supported model backbone is subject to its own licensing terms:

- JMP backbone: Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0) - [JMP License](https://github.com/facebookresearch/JMP/blob/main/LICENSE.md)
- EquiformerV2 backbone: Meta Research License - [EquiformerV2 License](https://huggingface.co/fairchem/OMAT24/blob/main/LICENSE)
- M3GNet backbone: BSD 3-Clause License - [M3GNet License](https://github.com/materialsvirtuallab/m3gnet/blob/main/LICENSE)
- ORB backbone: Apache License 2.0 - [ORB License](https://github.com/orbital-materials/orb-models/blob/main/LICENSE)

Please ensure compliance with the respective licenses when using specific model backbones in your project. For commercial use cases, carefully review each backbone's license terms or contact the respective authors for licensing options.

## Citation

If you use MatterTune in your research, please cite:

TODO
```bibtex
```
