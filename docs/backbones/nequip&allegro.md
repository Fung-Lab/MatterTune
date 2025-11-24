# Nequip/Allegro Backbone

The [NequIP and Allegro atomistic foundation models](https://www.nequip.net/) are equivariant interatomic potentials trained within the [NequIP](https://www.nature.com/articles/s41467-022-29939-5) and [Allegro](https://www.nature.com/articles/s41467-023-36329-y) model frameworks. Their key features lie in robustness and scalability improvements to the original NequIP framework, and in achieving faster inference through model packaging and compilation, as well as optimized tensor-product computations.

For more technical information, please refer to [paper](https://arxiv.org/abs/2504.16068)

## Installation

Both NequIP and Allegro models both depend on the ```nequip``` package. To setup the environment, please install ```nequip``` by following [nequip doc](https://nequip.readthedocs.io/en/latest/guide/getting-started/install.html).

Or the user can refer to the sample installation we provide:

```bash
conda create -n nequip-tune python=3.10 -y
conda activate nequip-tune
pip install torch
git clone --depth 1 https://github.com/mir-group/nequip.git
cd nequip
pip install .
```

## Models

There are currently 4 models available:
- [NequIP-OAM-L-0.1](https://www.nequip.net/models/mir-group/NequIP-OAM-L:0.1) (Featured Model)
- [NequIP-MP-L-0.1](https://www.nequip.net/models/mir-group/NequIP-MP-L:0.1)
- [Allegro-OAM-L-0.1](https://www.nequip.net/models/mir-group/Allegro-OAM-L:0.1) (Featured Model)
- [Allegro-MP-L-0.1](https://www.nequip.net/models/mir-group/Allegro-MP-L:0.1)


## License

The UMA backbone is available under MIT License

## Usage Example

### Fine-tuning

Please refer to our [example script](https://github.com/Fung-Lab/MatterTune/blob/add_nequip/examples/finetune-test/nequip_test.py)

### Model Packaging and Compiling

A suggested NequIP workflow is:
1. Train a NequIP model and save the checkpoint (.ckpt) file.
2. Test the trained model using the checkpoint file if needed.
3. Package the trained model into a NequIP package file (.nequip.zip).
4. Compile the NequIP package file into a compiled model file (.nequip.pth/pt2).

Since the MatterTune platform refactors the model implementations to satisfy its modular design principles, models trained on MatterTune cannot be directly packaged using the ```nequip-package``` API provided in the original NequIP repository. However, we have implemented an equivalent packaging function in MatterTune, ```mattertune.backbones.nequip_foundation.util.nequip_model_package()```, following the design of the nequip-package API. NequIP/Allegro models trained with MatterTune, once packaged using this function, are fully compatible with the other NequIP APIs, including ```nequip-compile```. 

An example:

```python
from mattertune.backbones.nequip_foundation.util import nequip_model_package

nequip_model_package(
    ckpt_path="YOUR_.ckpt_FILE_PATH",
    example_atoms=example_atoms, ## An example atoms for application
    output_path="YOUR_.nequip.zip_FILE_PATH",
)
```

The packaged model can be compiled by ```nequip-compile```

```bash
nequip-compile \
  YOUR_.nequip.zip_FILE_PATH \
  OUTPUT_FILE_PATH \
  --mode torchscript \
  --device cuda \
  --target ase
```

The compiled model can then be loaded for production simulations in supported integrations such as [nequip LAMMPS](https://nequip.readthedocs.io/en/latest/integrations/lammps/index.html) or [nequip ASE](https://nequip.readthedocs.io/en/latest/integrations/ase.html)