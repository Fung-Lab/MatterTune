from __future__ import annotations

import logging
from pathlib import Path
import rich
import os

from lightning.pytorch.strategies import DDPStrategy

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig
from mattertune.backbones import NequIPBackboneModule

logging.basicConfig(level=logging.ERROR)



def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()
        hparams.model = MC.NequIPBackboneConfig.draft()
        hparams.model.pretrained_model = "NequIP-OAM-L-0.1"
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
        hparams.data.pin_memory = False

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
        ckpt_name = "NequIP-OAM-best"
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
                name="NequIP",
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
    from ase.io import read
    from ase import Atoms
    import numpy as np
    import torch
    import wandb
    from tqdm import tqdm
    
    ckpt_path = "./checkpoints/NequIP-OAM-best.ckpt"
    model = NequIPBackboneModule.load_from_checkpoint(ckpt_path)
    
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
    
    ## Mixture of Experts (MoE) Inference Example
    from ase.md.langevin import Langevin
    from ase.io import read
    from ase import Atoms
    import time
    
    ckpt_path = "./checkpoints/NequIP-OAM-best.ckpt"
    model = NequIPBackboneModule.load_from_checkpoint(ckpt_path)
    
    ### before merging MoE
    atoms: Atoms = read("./data/Li_electrode_test.xyz", index=0) # type: ignore
    calc = model.ase_calculator(
        device = f"cuda:{args_dict['devices'][0]}"
    )
    atoms.set_calculator(calc)
    dyn = Langevin(
        atoms,
        timestep=1.0,
        temperature_K=300,
        friction=0.02,
    )
    time1 = time.time()
    dyn.run(1000) # 1000 steps
    time2 = time.time()
    rich.print(f"Inference Speed: {(time2 - time1)/1000} seconds/step")
    
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
