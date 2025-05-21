# MACE Backbone

MACE is a series of fast and accurate machine learning interatomic potentials with higher order equivariant message passing developed by Ilyes Batatia, Gregor Simm, David Kovacs, and the group of Gabor Csanyi in University of Cambridge. The MACE series released its first foundation model, [MACE-MP-0](https://arxiv.org/abs/2401.00096), in 2023, making it one of the earliest foundation models in the materials domain. To date, MACE has spawned several versions of its foundation models (see [MACE versions](https://github.com/ACEsuit/mace-foundations) for details) and has earned top marks on numerous leaderboards.

## Installation

MACE can be directly installed with pip:

```bash
pip install --upgrade pip
pip install mace-torch
```

or it can be installed from source code:

```bash
git clone https://github.com/ACEsuit/mace.git
pip install ./mace
```

## Key Features

MACE adopts an equivariant neural-network paradigm and delivers energy-conserving predictions of forces and stresses. For details on the specific features of each MACE version, please consult the introduction here: [MACE versions](https://github.com/ACEsuit/mace-foundations)

## License

The MatterSim backbone is available under MIT License