from __future__ import annotations

import logging
from pathlib import Path
import rich
from datetime import datetime
import os

import allegro.nn as allegro_nn
import nequip.nn.pair_potential as pair_potential

from lightning.pytorch.strategies import DDPStrategy

from mattertune.students.main import OfflineDistillationTrainerConfig, MatterTuneOfflineDistillationTrainer
from mattertune.configs import WandbLoggerConfig
import mattertune.configs as MC
from mattertune.students import AllegroStudentModel


def main(args_dict: dict):
    def hparams():
        hparams = OfflineDistillationTrainerConfig.draft()
        hparams.model = MC.AllegroStudentModelConfig.draft()
        hparams.model.model_dtype = "float64"
        hparams.model.r_max = args_dict["cutoff"]
        hparams.model.type_names = args_dict["elements"]
        hparams.model.l_max = 1
        hparams.model.parity = False
        hparams.model.two_body_mlp_hidden_layers_depth = 2
        hparams.model.two_body_mlp_hidden_layers_width = 64
        hparams.model.two_body_mlp_nonlinearity = "silu"
        hparams.model.radial_chemical_embed = {
            "_target_": allegro_nn.TwoBodyBesselScalarEmbed,
            "num_bessels": 8,
            "bessel_trainable": False,
            "polynomial_cutoff_p": 6,
        }
        hparams.model.radial_chemical_embed_dim = 64
        hparams.model.num_layers = args_dict["num_message_passing"]
        hparams.model.num_scalar_features = 1
        hparams.model.num_tensor_features = 1
        hparams.model.allegro_mlp_hidden_layers_depth = 1
        hparams.model.allegro_mlp_hidden_layers_width = 64
        hparams.model.allegro_mlp_nonlinearity = "silu"
        hparams.model.readout_mlp_hidden_layers_depth = 1
        hparams.model.readout_mlp_hidden_layers_width = 32
        hparams.model.readout_mlp_nonlinearity = None
        hparams.model.avg_num_neighbors = 12
        # hparams.model.per_type_energy_shifts = {
        #     "Li": -5.460104670490587,
        #     "P": -1.8200349348590048,
        #     "O": -7.280139642761157,
        # }
        
        hparams.model.optimizer = MC.AdamConfig(
            lr=args_dict["lr"],
        )
        hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
            mode="min",
            monitor=f"val/forces_mae",
            factor=0.5,
            patience=25,
            min_lr=1e-8,
        )
        
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(), loss_coefficient=1.0
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(), conservative=True, loss_coefficient=1.0
        )
        hparams.model.properties.append(forces)
        
        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.num_workers = 8
        hparams.data.train = MC.XYZDatasetConfig.draft()
        hparams.data.train.src = "/net/csefiles/coc-fung-cluster/lingyu/datasets/li3po4-train-10000.xyz"
        hparams.data.validation = MC.XYZDatasetConfig.draft()
        hparams.data.validation.src = "/net/csefiles/coc-fung-cluster/lingyu/datasets/li3po4-val-1000.xyz"
        hparams.data.batch_size = args_dict["batch_size"]
        
        ## Add Normalization for Energy
        hparams.model.normalizers = {
            "energy": [
                # MC.PerAtomReferencingNormalizerConfig(
                #     per_atom_references=Path("./data/li3po4_1593_eVAng-energy_reference.json")
                # ),
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
        hparams.trainer.gradient_clip_val = 100.0
        hparams.trainer.ema = MC.EMAConfig(decay=0.99)
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/forces_mae", patience=200, mode="min", min_delta=1e-5, verbose=True,
        )
        os.system(f"rm -rf ./checkpoints/allegro-{args_dict['cutoff']}A-T={args_dict['num_message_passing']}.ckpt")
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename=f"allegro-{args_dict['cutoff']}A-T={args_dict['num_message_passing']}",
            save_top_k=1,
            mode="min",
        )
        now = datetime.now()
        formatted = now.strftime("%m-%d %H:%M")
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Offline-Distill-Test",
                name=f"allegro-{args_dict['cutoff']}A-T={args_dict['num_message_passing']}-li3po4-{formatted}"
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
    
    val_atoms_list:list[Atoms] = read("/net/csefiles/coc-fung-cluster/lingyu/datasets/li3po4-val-1000.xyz", ":") # type: ignore
    
    model = AllegroStudentModel.load_from_checkpoint(f"./checkpoints/allegro-{args_dict['cutoff']}A-T={args_dict['num_message_passing']}.ckpt", weights_only=False)
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
    plt.savefig("allegro_li3po4_val_predictions.png", dpi=300)
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--num_message_passing", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--elements", type=str, nargs="+", default=["Li", "P", "O"])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
