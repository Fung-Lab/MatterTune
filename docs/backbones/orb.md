# ORB Backbone

The ORB backbone allows using pre-trained ORB models as a backbone in MatterTune. ORB is a library that provides pre-trained graph neural networks for predicting material properties.

## Configuration

To use an ORB backbone, specify the following configuration in your MatterTune config:

```python
@backbone_registry.register
class ORBBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["orb"] = "orb"
    """The type of the backbone."""

    pretrained_model: str
    """The name of the pretrained model to load."""

    system: ORBSystemConfig = ORBSystemConfig(radius=10.0, max_num_neighbors=20)
    """The system configuration, controlling how to featurize a system of atoms."""
```

### Key Parameters

- `pretrained_model`: The name of the pre-trained ORB model to load. Must be one of the models available in the `orb_models` package.
- `system`: Configures how to convert an `ase.Atoms` object into the graph representation expected by ORB:
  - `radius`: Specifies the cutoff distance for including neighbors in the graph
  - `max_num_neighbors`: Limits the maximum number of neighbors per node

## Supported Properties

The ORB backbone supports predicting the following properties:

- Energy (`EnergyPropertyConfig`)
- Forces (`ForcesPropertyConfig` with `conservative=False`)
- Stress (`StressesPropertyConfig` with `conservative=False`)
- Generic graph properties (`GraphPropertyConfig`)

```{note}
Conservative forces and stresses are currently not supported.
```

## Implementation Details

The key components of the ORB backbone implementation are:

### ORBBackboneModule
The main module that loads the pre-trained ORB model and defines the forward pass:
- `create_model`: Loads the pre-trained ORB model and initializes output heads
- `model_forward`: Runs the backbone model and output heads to predict properties
- `atoms_to_data`: Converts `ase.Atoms` objects to ORB graph representation

### Output Heads
Based on the property configuration, appropriate ORB output heads are initialized:
- `EnergyHead` for `EnergyPropertyConfig`
- `NodeHead` for `ForcesPropertyConfig`
- `GraphHead` for `StressesPropertyConfig` and `GraphPropertyConfig`

The ORB backbone leverages the `orb_models` and `nanoflann` packages to load pre-trained models and efficiently construct graph representations suitable for the ORB architecture.
