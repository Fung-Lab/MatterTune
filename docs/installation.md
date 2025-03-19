# Installation Guide

The installation of MatterTune consists of three stages:

1. Configure environment dependencies for one specific backbone model
2. Install the MatterTune package
3. Set up additional dependencies for external datasets and data sources

```{warning}
Since there are dependency conflicts between different backbone models, we strongly recommend creating separate virtual environments for each backbone model you plan to use.
```

## Backbone Installation

Below are the installation instructions for our currently supported backbone models using conda and pip.

### M3GNet

```bash
conda create -n matgl-tune python=3.10 -y
pip install matgl
pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

```{note}
Manual installation of `torch` and `dgl` packages after `matgl` installation is required to enable GPU acceleration for training.
```

### JMP

Please follow the installation instructions in the [jmp-backbone repository](https://github.com/nimashoghi/jmp-backbone/blob/lingyu-grad/README.md).

### ORB

ORB provides installation instructions in the [orb-models repository](https://github.com/orbital-materials/orb-models). However, we spotted some bugs in their implementation, which may cause bugs in multi-GPU training. So in this installation guidance, we suggest users to install from our pull request using the following instruction:

```bash
pip install "orb_models@git+https://github.com/nimashoghi/orb-models.git"
```

After installing ORB, we suggest users to check whether the urls of the pretrained checkpoints are in newest version. In ```orb_models/forcefield/pretrained.py```, checkpoint urls are stored with the parameter named as ```weighted_path```, which has been frequently changed recently. One quick solution is copy the newest version [here](https://github.com/orbital-materials/orb-models/blob/main/orb_models/forcefield/pretrained.py) and overwrite your local version. 

### EquiformerV2

```bash
conda create -n eqv2-tune python=3.10
conda activate eqv2-tune
pip install "git+https://github.com/FAIR-Chem/fairchem.git@omat24#subdirectory=packages/fairchem-core" --no-deps
pip install ase "e3nn>=0.5" hydra-core lmdb numba "numpy>=1.26,<2.0" orjson "pymatgen>=2023.10.3" submitit tensorboard "torch==2.5.0" wandb torch_geometric h5py netcdf4 opt-einsum spglib
```

### MatterSim

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

## MatterTune Package Installation

```{important}
MatterTune should be installed after setting up the backbone model dependencies.
```

Clone the repository and install MatterTune by:

```bash
pip install -e .
```

## External Dataset Installation

### Matbench

Clone the repo and install by:
```bash
git clone https://github.com/hackingmaterials/matbench
cd matbench
pip install -e . -r requirements-dev.txt
```

### Materials Project

Install mp-api:
```bash
pip install mp-api
```

```{note}
There are currently dependency conflicts between mp-api and matbench packages. You may not be able to use both simultaneously in a single virtual environment.
```

### Materials Project Trajectories

To access MPTraj data from our Hugging Face dataset:
```bash
pip install datasets
```
