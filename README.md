# MatterTune: A Unified Platform for Atomistic Foundation Model Fine-Tuning

## Table of Contents


## About MatterTune

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

## Supported Models

MatterTune currently provides support for the following Atomistic Foundation Models:

- JMP
- EquiformerV2
- Orb
- M3GNet (MatGL repository implementation)

> Here add a brief description and link to each model?

MatterTune is still under active development and maintenance. We are continuously working to expand support for more mainstream open-source Atomistic Foundation Models. We welcome collaboration with other model development teams to help integrate their models into the MatterTune platform.

## Supported Downstream Task Interfaces

MatterTune currently provides support for ```ase```:
- For any model trained on MatterTune, it can be automatically wrapped into a ```MatterTunePotential``` class, which can make prediction on a list of ```ase.Atoms``` and can be accelerated by cpu, single gpu, or multi-gpus.
- For any model trained on MatterTune, it can also be automatically wrapped into a ```ase.calculators.calculator.Calculator```, which can set as the calculator of ```ase.Atoms``` for property prediction, structure optimization, molecular dynamics, and so on.

## Installation

The setup process for using MatterTune consists of two steps:

1. Installing model-specific dependencies
2. Installing MatterTune on top of these dependencies

Below are the detailed installation instructions for each supported model:

> Provide how to install each model's env and test it.

## MatterTune in 15 minutes.

This section outlines the fundamental workflow for model training and accessing downstream task interfaces in MatterTune. For detailed parameter configurations and method specifications, please refer to our CoreAPI, OtherAPI and Glossary sections.

### Training

A MatterTune training task is configured through `MC.MatterTunerConfig`. It can be declared by:

```python
hparams = MC.MatterTunerConfig.draft()
```

Then a then training task can be configured and started through for steps:

#### Step1. Model Configuration (`MC.MatterTunerConfig.model`)
    
```python
# Set backbone model (JMP in this example)
hparams.model = MC.JMPBackboneConfig.draft()
hparams.model.graph_computer = MC.JMPGraphComputerConfig.draft()
hparams.model.graph_computer.pbc = True  # For periodic structures

# Specify checkpoint path
hparams.model.ckpt_path = Path("./jmp-s.pt")

# Configure optimizer
hparams.model.optimizer = MC.AdamWConfig(lr=8e-5)

# Define training properties and loss functions
hparams.model.properties = [
    MC.EnergyPropertyConfig(loss=MC.MAELossConfig(), loss_coefficient=1.0),
    MC.ForcesPropertyConfig(loss=MC.MAELossConfig(), loss_coefficient=10.0, conservative=False)
]
```
The model configuration specifies the backbone model type (e.g., JMP, or other supported models), optimizer settings, and target properties.

#### Step2. **Data Configuration** (`MC.MatterTunerConfig.data`)

```python
# Configure data loading
hparams.data = MC.AutoSplitDataModuleConfig.draft()
hparams.data.dataset = MC.XYZDatasetConfig.draft()
hparams.data.dataset.src = Path("dataset.xyz")
hparams.data.train_split = 0.8
hparams.data.batch_size = 64
```

MatterTune supports multiple data formats including ```.xyz```, ```ase.db```, MPTraj, OMat24, Matbench, and so on. It also supports auto-split and manual-split to suits various scenarios. 

#### Step3. **Trainer Settings** (`MC.MatterTunerConfig.lightning_trainer_kwargs`)

```python
# Trainer Configuration
from lightning.pytorch.strategies import DDPStrategy
    
hparams.lightning_trainer_kwargs = {
    "max_epochs": 100,
    "accelerator": "gpu",
    "devices": [0, 1],
    "strategy": DDPStrategy(find_unused_parameters=True),
}
```

The ```hparams.lightning_trainer_kwargs``` exactly uses ```lightning.Trainer``` parameters, supporting features like multi-GPU training.

#### Step4. Start Training

Once you have prepared your configuration (let's call it `hparams`), you can start the training process with just two lines of code:

```python
mt_config = hparams.finalize()
model = MatterTuner(mt_config).tune()
```

### Accessing Model Interfaces

After successful training, you can easily wrap your model into both a potential and an ASE calculator, 

```python
# Create a potential for batch predictions
potential = model.potential(lightning_trainer_kwargs=None)

# Create an ASE calculator for atomic simulations
calculator = model.ase_calculator(lightning_trainer_kwargs=None)
```

> Note: The lightning_trainer_kwargs parameter allows customization of prediction behavior. When set to None, it uses default PyTorch Lightning trainer settings.

These interfaces can be directly used in downstream tasks. For example:

```python
# Batch prediction using potential
atoms1 = ...  # Your first atomic structure
atoms2 = ...  # Your second atomic structure
predictions = potential.predict([atoms1, atoms2])

# Using ASE calculator for single structure calculations
atoms1.calc = calculator
energy = atoms.get_potential_energy() ## if energy property specified in target during training
bandgap = atoms.get_property(["bandgap"]) ## if "bandgap" property specified in target during training
```

## Core API

Unfinished, to be listed here:
- MC
- MC.MatterTunerConfig
- MC.backbone configs
- MC.properties
- MC.data...
- MatterTuner

## Other API

Unfinished, to be listed here:
- Optimizer
- learning rate scheduler
- losses

## Tutorials

### Quick Start
For quick start and general usage examples, explore the interactive notebooks in the `notebooks` directory.

### Comprehensive Examples
The `examples` directory contains three detailed case studies demonstrating MatterTune's capabilities:

1. **Water Thermodynamics Study**
  - Fine-tuning pretrained models on water thermodynamics datasets
  - Performing ice-melting NVT molecular dynamics simulations using the fine-tuned model

2. **Crystal Structure Optimization**
  - Fine-tuning pretrained models on Zn-Mn-O subset from MPTraj dataset
  - Structure relaxation of random ZnMn<sub>2</sub>O<sub>4</sub> configurations

3. **Materials Property Screening**
  - Fine-tuning pretrained models on Matbench tasks
  - High-throughput property screening applications

Each example includes detailed documentation and scripts to help you adapt these workflows to your specific research needs.

## Contact
missing

## License
missing
