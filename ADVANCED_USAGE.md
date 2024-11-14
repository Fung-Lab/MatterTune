# MatterTune Advanced Usage Guide

This guide covers advanced topics for extending MatterTune with custom components. You'll learn how to create custom backbones (model architectures) and datasets.

## Table of Contents
- [MatterTune Advanced Usage Guide](#mattertune-advanced-usage-guide)
    - [Table of Contents](#table-of-contents)
    - [Implementing Custom Backbones](#implementing-custom-backbones)
        - [Basic Structure](#basic-structure)
        - [Required Methods](#required-methods)
    - [Implementing Custom Datasets](#implementing-custom-datasets)
        - [Dataset Structure](#dataset-structure)
    - [Usage](#usage)
    - [Best Practices](#best-practices)

## Implementing Custom Backbones

### Basic Structure
To implement a custom backbone, you need to create two classes:
1. A configuration class inheriting from `FinetuneModuleBaseConfig`
2. A model class inheriting from `FinetuneModuleBase`

Here's the basic template:

```python
from typing import Literal
from typing_extensions import override
import mattertune as mt

@mt.backbone_registry.register
class MyBackboneConfig(mt.FinetuneModuleBaseConfig):
    name: Literal["my_backbone"] = "my_backbone"

    # Add your configuration parameters
    hidden_size: int
    num_layers: int

    @override
    @classmethod
    def model_cls(cls):
        return MyBackboneModule

class MyBackboneModule(mt.FinetuneModuleBase["MyData", "MyBatch", MyBackboneConfig]):
    @override
    @classmethod
    def hparams_cls(cls):
        return MyBackboneConfig

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Check for required packages
        pass
```

### Required Methods

Your backbone module must implement these abstract methods:

```python
class MyBackboneModule(mt.FinetuneModuleBase):
    @override
    def create_model(self):
        """Initialize your model architecture here"""
        pass

    @override
    def model_forward(self, batch, return_backbone_output=False):
        """Forward pass implementation"""
        pass

    @override
    def pretrained_backbone_parameters(self):
        """Return backbone parameters"""
        pass

    @override
    def output_head_parameters(self):
        """Return output head parameters"""
        pass

    @override
    def cpu_data_transform(self, data):
        """Transform data on CPU before batching"""
        pass

    @override
    def collate_fn(self, data_list):
        """Combine data samples into a batch"""
        pass

    @override
    def gpu_batch_transform(self, batch):
        """Transform batch on GPU before forward pass"""
        pass

    @override
    def batch_to_labels(self, batch):
        """Extract ground truth labels from batch"""
        pass

    @override
    def atoms_to_data(self, atoms, has_labels):
        """Convert ASE Atoms to your data format"""
        pass

    @override
    def create_normalization_context_from_batch(self, batch):
        """Create context for property normalization"""
        pass
```

For detailed examples of backbone implementations, please refer to the existing backbone implementations in the MatterTune source code:
- JMP backbone: `mattertune/backbones/jmp/model.py`
- EquiformerV2 backbone: `mattertune/backbones/eqV2/model.py`
- M3GNet backbone: `mattertune/backbones/m3gnet/model.py`
- ORB backbone: `mattertune/backbones/orb/model.py`

## Implementing Custom Datasets

### Dataset Structure

Custom datasets require two classes:
1. A configuration class inheriting from `DatasetConfigBase`
2. A dataset class inheriting from `Dataset[ase.Atoms]`

Basic template:

```python
from typing import Literal
from typing_extensions import override
import mattertune as mt
from torch.utils.data import Dataset
from ase import Atoms

@mt.data_registry.register
class MyDatasetConfig(mt.DatasetConfigBase):
    type: Literal["my_dataset"] = "my_dataset"

    # Add your configuration parameters
    data_path: str

    @override
    def create_dataset(self):
        return MyDataset(self)

class MyDataset(Dataset[Atoms]):
    def __init__(self, config: MyDatasetConfig):
        self.config = config
        # Initialize your dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Atoms:
        # Return an ASE Atoms object
        return self.data[idx]
```

For detailed examples of dataset implementations, please refer to the existing dataset implementations in the MatterTune source code:
- XYZ dataset: `mattertune/data/xyz.py`
- ASE database dataset: `mattertune/data/db.py`
- Materials Project dataset: `mattertune/data/mp.py`
- Matbench dataset: `mattertune/data/matbench.py`

## Usage

After implementing your custom components, you can use them in your training configuration:

```python
config = mt.configs.MatterTunerConfig(
    model=MyBackboneConfig(
        hidden_size=256,
        num_layers=4,
        properties=[
            mt.configs.EnergyPropertyConfig(...)
        ]
    ),
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=MyDatasetConfig(
            data_path="path/to/data"
        ),
        train_split=0.8,
        batch_size=32
    )
)

tuner = mt.MatterTuner(config)
model, trainer = tuner.tune()
```

## Best Practices

1. **Type Hints**: Always use proper type hints to catch errors early
2. **Documentation**: Document your custom components thoroughly
3. **Error Handling**: Provide clear error messages for configuration issues
4. **Testing**: Write tests for your custom components
5. **Dependencies**: Clearly document any additional dependencies

For more examples, check the source code of the built-in backbones and datasets in the MatterTune repository.
