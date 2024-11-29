# Motivation

## The Problem

The emergence of atomistic foundation models has created a pressing need for standardized fine-tuning frameworks. However, several key challenges have made it difficult to create a unified platform:

1. **Diverse Model Architectures**: Atomistic models use various architectural paradigms (GNNs, Transformers) with different internal representations (PyG graphs, DGL graphs, dense tensors).

2. **Different Data Processing Requirements**: Each model requires specific data preprocessing pipelines and batch structures, making standardization challenging.

3. **Complex Property Prediction**: Models must handle diverse property types (scalar, vector, per-atom, system-level) with varying output head architectures.

4. **Integration Complexity**: Models need to interface with existing molecular dynamics engines, structure prediction software, and materials screening pipelines.

## Core Design Philosophy

MatterTune addresses these challenges through a carefully designed abstraction hierarchy that maximizes flexibility while maintaining a clean, unified interface:

### 1. Data Abstraction

The foundation of MatterTune is a minimalist data contract, allowing for support for any data source that can provide atomic structures as ASE Atoms objects:

```python
import ase
from torch.utils.data import Dataset

class MyDataset(Dataset[ase.Atoms]):
    """A dataset that provides atomic structures.

    This is the minimal interface required by MatterTune. Any data source that can be
    mapped to ASE Atoms objects can be wrapped in this interface.
    """

    def __init__(self, data_source: str):
        """Initialize the dataset.

        Args:
            data_source: Path to data or other source identifier
        """
        self.data = ...  # Load your data

    def __len__(self) -> int:
        """Return the number of structures in the dataset."""
        return len(self.data)

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        """Return the atomic structure at given index.

        Args:
            idx: Index of the desired structure

        Returns:
            ase.Atoms: The atomic structure
        """
        return self.data[idx]
```

This simple abstraction enables support for any data source that can provide atomic structures, providing several key benefits:
- Universal compatibility with existing materials science formats
- Zero assumptions about internal data storage
- Natural integration with ASE's ecosystem
- Flexibility to support any data source that can be mapped to atomic structures


### 2. Backbone Abstraction

Rather than enforcing a specific internal architecture, MatterTune defines backbones through their capability to predict properties:

```python
class ModelOutput(TypedDict):
    predicted_properties: dict[str, torch.Tensor]
    """Predicted properties. This dictionary should be exactly
        in the same shape/format  as the output of `batch_to_labels`."""

    backbone_output: NotRequired[Any]
    """Output of the backbone model. Only set if `return_backbone_output` is True."""

class FinetuneModuleBase(Generic[TData, TBatch]):
    @abstractmethod
    def atoms_to_data(self, atoms: Atoms, has_labels: bool) -> TData:
        """Convert atoms to model-specific data format"""

    @abstractmethod
    def collate_fn(self, data_list: list[TData]) -> TBatch:
        """Collate individual data points into a batch"""

    @abstractmethod
    def model_forward(self, batch: TBatch) -> ModelOutput:
        """Predict properties from a batch"""
```

This design:
- Allows models to use their native data structures (TData, TBatch)
- Separates property schema from implementation details
- Enables efficient batch processing specific to each architecture
- Provides clear extension points for new model types

### 3. Property Schema

Properties are defined through a declarative schema system:

```python
class EnergyPropertyConfig:
    """Configuration for total energy prediction."""
    name: str = "energy"  # Fixed name for energy property
    loss: LossConfig      # Loss function configuration
    loss_coefficient: float = 1.0  # Weight in total loss

class ForcesPropertyConfig:
    """Configuration for atomic forces prediction."""
    name: str = "forces"
    loss: LossConfig
    loss_coefficient: float = 1.0
    conservative: bool  # Whether forces are computed as energy gradients

class StressesPropertyConfig:
    """Configuration for stress tensor prediction."""
    name: str = "stress"
    loss: LossConfig
    loss_coefficient: float = 1.0
    conservative: bool  # Whether stress is computed from energy

class GraphPropertyConfig:
    """Configuration for custom graph-level properties."""
    name: str  # User-defined property name
    loss: LossConfig
    loss_coefficient: float = 1.0
    reduction: Literal["mean", "sum", "max"]  # How to aggregate atomic features

PropertyConfig = TypeAliasType("PropertyConfig", EnergyPropertyConfig | ForcesPropertyConfig | StressesPropertyConfig | GraphPropertyConfig)
```

Benefits:
- Clear separation between property definition and implementation
- Type-safe property specifications
- Provides built-in support for common properties (energy, forces, stress)
- Support for complex property types
- Flexible reduction strategies

## Implementation Philosophy

The framework follows several key principles:

1. **Minimal Assumptions**: We make zero assumptions about internal model architectures or data structures beyond the basic interfaces.

2. **Type Safety**: All interfaces are fully typed, providing clear contracts and early error detection.

3. **Separation of Concerns**:
   - Property definitions are separate from implementations
   - Data processing is separate from model architecture
   - Training logic is separate from model definition

4. **Extensibility First**:
   - New backbones only need to implement core data conversion methods
   - Custom datasets only need to map to ase.Atoms
   - Property types can be extended without changing the core framework

## Real-World Benefits

This design enables several powerful workflows:

1. **Unified Fine-tuning**: Train any supported model on any compatible dataset with a consistent API.

2. **Easy Integration**: Models automatically work with ASE calculators and MD engines.

3. **Flexible Deployment**: Models can be used for:
   - Molecular dynamics simulations
   - High-throughput screening
   - Structure prediction
   - Property prediction

4. **Performance Optimization**: Each model can implement optimal batch processing while maintaining a consistent interface.

## Future Extensibility

The framework is designed to grow with the field:

1. **New Architectures**: Additional backbones can be added by implementing the core interfaces.

2. **New Properties**: The property schema system can be extended for new property types.

3. **New Data Sources**: Any data source that can map to ase.Atoms is supported.

4. **New Applications**: The clean interfaces enable integration with new workflows and tools.
