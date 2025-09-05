# MACE Backbone

"UMA (Universal Models for Atoms) is a next-generation atomistic foundation model developed by Meta FAIRChem team. The UMA models are based on the [eSEN](https://arxiv.org/abs/2502.12147) architecture and trained on 5 billion unique 3D atomic structures — the largest training run to date — by compiling data from multiple chemical domains, including molecules, materials, catalysts, and metal–organic frameworks. In addition, UMA introduces a mixture-of-experts design, which scales the parameter size through the combination of multiple expert models. When performing structure optimization or molecular dynamics simulations, these expert models can be integrated to achieve the same speed as a single expert model.

## Installation

To use the UMA series models, please first make sure you have obtained Hugging Face access at https://huggingface.co/facebook/UMA in order to access their checkpoints and created a Huggingface token at https://huggingface.co/settings/tokens/ with this permission:
- Permissions: Read access to contents of all public gated repos you can access

Then, add the token as an environment variable

```bash 
# pip install huggingface_hub
huggingface-cli login
```

It may be enough to use ```pip install fairchem-core``` to install UMA. This gets you the latest version on PyPi (https://pypi.org/project/fairchem-core/)

Or you can install from source code in edit mode:

```bash
git clone git@github.com:facebookresearch/fairchem.git
pip install -e fairchem/packages/fairchem-core
# or 
# pip install -e fairchem/packages/fairchem-core[dev] 
# for development
```

## Key Features

We consider the key feature of the UMA model to be its unprecedented breadth of training domains among atomistic foundation models to date, encompassing molecules, materials, catalysts, and metal–organic frameworks. For detailed information about UMA, please see [UMA: A Family of Universal Models for Atoms](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/)

## License

The UMA backbone is available under MIT License

## Usage Example

### UMA Fine-tuning

```python
from __future__ import annotations

import logging
from pathlib import Path
import rich
import os

from lightning.pytorch.strategies import DDPStrategy

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig
from mattertune.backbones.uma import UMABackboneModule

logging.basicConfig(level=logging.ERROR)



def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()
        hparams.model = MC.UMABackboneConfig.draft()
        hparams.model.model_name = "uma-s-1.1"
        hparams.model.task_name = "omat"
        hparams.model.reset_output_heads = True
        hparams.model.optimizer = MC.AdamWConfig(
            lr=args_dict["lr"],
            amsgrad=False,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.1,
        )
        hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
            mode="min",
            monitor=f"val/forces_mae",
            factor=0.8,
            patience=5,
            min_lr=1e-8,
        )
        
        # Add model properties
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(), loss_coefficient=1.0
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(), conservative=True, loss_coefficient=1.0
        )
        hparams.model.properties.append(forces)
        stress = MC.StressesPropertyConfig(
            loss=MC.MSELossConfig(), conservative=True, loss_coefficient=1.0
        )
        hparams.model.properties.append(stress)

        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.train = MC.XYZDatasetConfig.draft()
        hparams.data.train.src = "./data/Li_electrode_finetune.xyz"
        hparams.data.validation = MC.XYZDatasetConfig.draft()
        hparams.data.validation.src = "./data/Li_electrode_val.xyz"
        hparams.data.batch_size = args_dict["batch_size"]

        ## Add Normalization for Energy
        hparams.model.normalizers = {
            "energy": [
                MC.PerAtomNormalizerConfig(),
            ]
        }
        
        ## Configure EMA
        hparams.trainer.ema = MC.EMAConfig(decay=0.99)

        ## Trainer Hyperparameters
        hparams.trainer = MC.TrainerConfig.draft()
        hparams.trainer.max_epochs = 20
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.strategy = DDPStrategy()
        hparams.trainer.gradient_clip_algorithm = "norm"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "32"

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/forces_mae", patience=50, mode="min", min_delta=1e-4
        )

        # Configure Model Checkpoint
        ckpt_name = "uma-s11-best"
        if os.path.exists(f"./checkpoints/{ckpt_name}.ckpt"):
            os.remove(f"./checkpoints/{ckpt_name}.ckpt")
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename=ckpt_name,
            save_top_k=1,
            mode="min",
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-UsageTest", 
                name="UMA",
            )
        ]

        # Additional trainer settings
        hparams.trainer.additional_trainer_kwargs = {
            "inference_mode": False,
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    model, trainer = MatterTuner(mt_config).tune()
    
    
    ## Perform Evaluation

    ckpt_path = "./checkpoints/uma-s11-best.ckpt"
    model = UMABackboneModule.load_from_checkpoint(ckpt_path)
    
    from ase.io import read
    from ase import Atoms
    import numpy as np
    import torch
    import wandb
    from tqdm import tqdm
    
    wandb.init(project="MatterTune-UsageTest", name="UMA", resume=True)
    
    val_atoms_list:list[Atoms] = read("./data/Li_electrode_test.xyz", ":") # type: ignore
    calc = model.ase_calculator(
        device = f"cuda:{args_dict['devices'][0]}"
    )
    energies_per_atom = []
    forces = []
    stresses = []
    pred_energies_per_atom = []
    pred_forces = []
    pred_stresses = []
    for atoms in tqdm(val_atoms_list):
        energies_per_atom.append(atoms.get_potential_energy() / len(atoms))
        forces.extend(np.array(atoms.get_forces()).tolist())
        stresses.extend(np.array(atoms.get_stress(voigt=False)).tolist())
        atoms.set_calculator(calc)
        pred_energies_per_atom.append(atoms.get_potential_energy() / len(atoms))
        pred_forces.extend(np.array(atoms.get_forces()).tolist())
        pred_stresses.extend(np.array(atoms.get_stress(voigt=False)).tolist())
        
    e_mae = torch.nn.L1Loss()(torch.tensor(energies_per_atom), torch.tensor(pred_energies_per_atom))
    f_mae = torch.nn.L1Loss()(torch.tensor(forces), torch.tensor(pred_forces))
    s_mae = torch.nn.L1Loss()(torch.tensor(stresses), torch.tensor(pred_stresses))
    
    rich.print(f"Energy MAE: {e_mae} eV/atom")
    rich.print(f"Forces MAE: {f_mae} eV/Ang")
    rich.print(f"Stresses MAE: {s_mae} eV/Ang^3")
    
    wandb.finish()
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--devices", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)

```

### Mixture of Experts Merging

If during your whole task, the overall charge and spin tag of your structure, as well as the element of each atom, are all fixed, you can accelerate UMA's inferencing by ```model.merge_MOLE_model(atoms)```. An example is:

```python
from ase.md.langevin import Langevin
from ase.io import read
from ase import Atoms
from mattertune.backbones.uma import UMABackboneModule
    
ckpt_path = "./checkpoints/uma-s11-best.ckpt"
model = UMABackboneModule.load_from_checkpoint(ckpt_path)
atoms: Atoms = read("./data/Li_electrode_test.xyz", index=0) # type: ignore
model.merge_MOLE_model(atoms)
calc = model.ase_calculator(
    device = f"cuda:0"
)
    
atoms.set_calculator(calc)
dyn = Langevin(
    atoms,
    timestep=1.0,
    temperature_K=300,
    friction=0.02,
)
dyn.run(1000)
```