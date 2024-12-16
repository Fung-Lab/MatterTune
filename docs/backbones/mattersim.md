# MatterSim Backbone

> Note: As of the latest MatterTune update, MatterSim has only released the M3GNet model.

The MatterSim backbone integrates the MatterSim model architecture into MatterTune. MatterSim is a foundational atomistic model designed to simulate materials property across wide range of elements, temperatures and pressures. 

## Installation

We strongly recommand to install MatterSim from source code

```bash
git clone git@github.com:microsoft/mattersim.git
cd mattersim
```

Find the line 41 of the pyproject.toml in MatterSim, which is ```"pydantic==2.9.2",```. Change it to ```"pydantic>=2.9.2",```. After finishing this modification, install MatterSim by running:

```bash
mamba env create -f environment.yaml
mamba activate mattersim
uv pip install -e .
python setup.py build_ext --inplace
```

## Key Features

- Pretrained on materials data across wide range of elements, temperatures and pressures.
- Flexible model architecture selection
  - MatterSim-v1.0.0-1M: A mini version of the M3GNet that is faster to run.
  - MatterSim-v1.0.0-5M: A larger version of the M3GNet that is more accurate.
  - TO BE RELEASED: Graphormer model with even larger parameter scale
- Support for property predictions:
  - Energy (extensive/intensive)
  - Forces (conservative for M3GNet and non-conservative for Graphormer)
  - Stresses (conservative for M3GNet and non-conservative for Graphormer)
  - Graph-level properties (available on Graphormer)

## Configuration

Here's a complete example showing how to configure the JMP backbone:

```python
from mattertune import configs as MC
from pathlib import Path

config = MC.MatterTunerConfig(
    model=MC.MatterSimBackboneConfig(
        # Required: Path to pre-trained checkpoint
        pretrained_model="MatterSim-v1.0.0-5M",

        # Graph construction settings
        graph_convertor=MC.MatterSimGraphConvertorConfig(
            twobody_cutoff = 5.0 ## The cutoff distance for the two-body interactions.
            has_threebody = True ## Whether to include three-body interactions.
            threebody_cutoff = 4.0 ## The cutoff distance for the three-body interactions.
        )

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

## Examples & Notebooks

A notebook tutorial about how to fine-tune and use MatterSim model can be found in ```notebooks/mattersim-waterthermo.ipynb```([link](https://github.com/Fung-Lab/MatterTune/blob/main/notebooks/mattersim-waterthermo.ipynb)). 

Under ```water-thermodynamics```([link](https://github.com/Fung-Lab/MatterTune/tree/main/examples/water-thermodynamics)), we gave an advanced usage example fine-tuning MatterSim on PES data and applying to MD simulation

## License

The MatterSim backbone is available under MIT License