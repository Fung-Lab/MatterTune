# Model Usage Guide

After training, you can use your model for predictions in several ways. This guide covers loading models and making predictions.

## Loading a Model

To load a saved model checkpoint (from a previous fine-tuning run), use the `load_from_checkpoint` method:

```python
from mattertune.backbones import JMPBackboneModule

model = JMPBackboneModule.load_from_checkpoint("path/to/checkpoint.ckpt")
```

## Making Predictions

The `MatterTunePropertyPredictor` interface provides a simple way to make predictions for a single or batch of atoms:

```python
from ase import Atoms
import torch

# Create ASE Atoms objects
atoms1 = Atoms('H2O',
              positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
              cell=[10, 10, 10],
              pbc=True)

atoms2 = Atoms('H2O',
                positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                cell=[10, 10, 10],
                pbc=True)

# Get predictions using the model's property predictor interface
property_predictor = model.property_predictor()
predictions = property_predictor.predict([atoms1, atoms2], ["energy", "forces"])

print("Energy:", predictions[0]["energy"], predictions[1]["energy"])
print("Forces:", predictions[0]["forces"], predictions[1]["forces"])
```

## Using as ASE Calculator

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
