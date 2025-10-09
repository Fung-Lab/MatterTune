from __future__ import annotations

import logging
from pathlib import Path
import rich
from datetime import datetime
import os

from lightning.pytorch.strategies import DDPStrategy

from mattertune.students.main import OfflineDistillationTrainerConfig, MatterTuneOfflineDistillationTrainer
from mattertune.configs import WandbLoggerConfig
import mattertune.configs as MC
from mattertune.students import CACEStudentModel


def main(args_dict: dict):
    def hparams():
        hparams = OfflineDistillationTrainerConfig.draft()
        hparams.model = MC.CACEStudentModelConfig.draft()
        hparams.model.zs = [1, 8]
        hparams.model.n_atom_basis = 3
        hparams.model.cutoff = 5.5
        hparams.model.cutoff_fn = MC.CACECutoffFnConfig(fn_type="polynomial", p=6)
        hparams.model.radial_basis = MC.CACERBFConfig(rbf_type="bessel", n_rbf=6, trainable=True)
        hparams.model.max_l = 3
        hparams.model.max_nu = 3
        hparams.model.n_radial_basis = 12
        hparams.model.num_message_passing = 1
        hparams.model.avg_num_neighbors = 1
        hparams.model.type_message_passing = ["Bchi"]
        hparams.model.args_message_passing = {'Bchi': {'shared_channels': False, 'shared_l': False}}
        
        hparams.model.optimizer = MC.AdamConfig(
            lr=args_dict["lr"],
        )
        hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
            mode="min",
            monitor=f"val/forces_rmse",
            factor=0.8,
            patience=10,
            min_lr=1e-8,
        )
        
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(), loss_coefficient=1.0
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(), conservative=True, loss_coefficient=1000.0
        )
        hparams.model.properties.append(forces)
        
        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.num_workers = 8
        hparams.data.train = MC.XYZDatasetConfig.draft()
        hparams.data.train.src = "./data/train_water_1593_eVAng.xyz"
        hparams.data.validation = MC.XYZDatasetConfig.draft()
        hparams.data.validation.src = "./data/val_water_1593_eVAng.xyz"
        hparams.data.batch_size = args_dict["batch_size"]
        
        ## Add Normalization for Energy
        hparams.model.normalizers = {
            "energy": [
                MC.PerAtomReferencingNormalizerConfig(
                    per_atom_references=Path("./data/water_1593_eVAng-energy_reference.json")
                ),
                MC.PerAtomNormalizerConfig(),
            ]
        }
        
        ## Trainer Hyperparameters
        hparams.trainer = MC.TrainerConfig.draft()
        hparams.trainer.max_epochs = args_dict["max_epochs"]
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.strategy = DDPStrategy()
        hparams.trainer.precision = "32"
        hparams.trainer.gradient_clip_algorithm = "value"
        hparams.trainer.gradient_clip_val = 10.0
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/forces_rmse", patience=200, mode="min"
        )
        os.system("rm -rf ./checkpoints/cace-best.ckpt")
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_rmse",
            dirpath="./checkpoints",
            filename="cace-best",
            save_top_k=1,
            mode="min",
        )
        now = datetime.now()
        formatted = now.strftime("%m-%d %H:%M")
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Offline-Distill-Test",
                name=f"cace-water-{formatted}"
            )
        ]
        hparams.trainer.additional_trainer_kwargs = {
            "inference_mode": False,
        }
        
        hparams = hparams.finalize(strict=False)
        return hparams
    
    train_config = hparams()
    model = MatterTuneOfflineDistillationTrainer(train_config).train()
    
    from ase.io import read
    from ase import Atoms
    import numpy as np
    import torch
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    val_atoms_list:list[Atoms] = read("./data/val_water_1593_eVAng.xyz", ":") # type: ignore
    
    model = CACEStudentModel.load_from_checkpoint("./checkpoints/cace-best.ckpt", lazy_init_atoms=val_atoms_list[0])
    calc = model.ase_calculator(device=f"cuda:{args_dict['devices'][0]}")
    
    energies = []
    energies_per_atom = []
    forces = []
    pred_energies = []
    pred_energies_per_atom = []
    pred_forces = []
    for atoms in tqdm(val_atoms_list):
        energies.append(atoms.get_potential_energy())
        energies_per_atom.append(atoms.get_potential_energy() / len(atoms))
        forces.extend(np.array(atoms.get_forces()).tolist())
        atoms.calc = calc
        pred_energies.append(atoms.get_potential_energy())
        pred_energies_per_atom.append(atoms.get_potential_energy() / len(atoms))
        pred_forces.extend(np.array(atoms.get_forces()).tolist())
        
    e_mae = torch.nn.L1Loss()(torch.tensor(energies_per_atom), torch.tensor(pred_energies_per_atom))
    f_mae = torch.nn.L1Loss()(torch.tensor(forces), torch.tensor(pred_forces))
    e_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(energies_per_atom), torch.tensor(pred_energies_per_atom)))
    f_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(forces), torch.tensor(pred_forces)))
    
    rich.print(f"Energy MAE: {e_mae} eV/atom")
    rich.print(f"Forces MAE: {f_mae} eV/Ang")
    rich.print(f"Energy RMSE: {e_rmse} eV/atom")
    rich.print(f"Forces RMSE: {f_rmse} eV/Ang")
    
    energies = np.array(energies)
    pred_energies = np.array(pred_energies)
    forces = np.array(forces).reshape(-1, 3)
    pred_forces = np.array(pred_forces).reshape(-1, 3)
    
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.scatter(energies, pred_energies, alpha=0.5)
    plt.xlabel("DFT Energy (eV)")
    plt.ylabel("Predicted Energy (eV)")
    plt.title("Energy Prediction")
    plt.plot([min(energies), max(energies)], [min(energies), max(energies)], 'r--')
    plt.subplot(2, 2, 2)
    plt.scatter(forces[:, 0], pred_forces[:, 0], alpha=0.5)
    plt.xlabel("DFT Forces X (eV/Ang)")
    plt.ylabel("Predicted Forces X (eV/Ang)")
    plt.title("Forces X Prediction")
    plt.plot([min(forces[:, 0]), max(forces[:, 0])], [min(forces[:, 0]), max(forces[:, 0])], 'r--')
    plt.subplot(2, 2, 3)
    plt.scatter(forces[:, 1], pred_forces[:, 1], alpha=0.5)
    plt.xlabel("DFT Forces Y (eV/Ang)")
    plt.ylabel("Predicted Forces Y (eV/Ang)")
    plt.title("Forces Y Prediction")
    plt.plot([min(forces[:, 1]), max(forces[:, 1])], [min(forces[:, 1]), max(forces[:, 1])], 'r--')
    plt.subplot(2, 2, 4)
    plt.scatter(forces[:, 2], pred_forces[:, 2], alpha=0.5)
    plt.xlabel("DFT Forces Z (eV/Ang)")
    plt.ylabel("Predicted Forces Z (eV/Ang)")
    plt.title("Forces Z Prediction")
    plt.plot([min(forces[:, 2]), max(forces[:, 2])], [min(forces[:, 2]), max(forces[:, 2])], 'r--')
    plt.tight_layout()
    plt.savefig("cace_water_val_predictions.png", dpi=300)
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
