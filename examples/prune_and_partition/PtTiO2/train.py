from __future__ import annotations

import logging
from pathlib import Path
import rich
import os

import nshutils as nu
from lightning.pytorch.strategies import DDPStrategy

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig
from mattertune.backbones.jmp.model import get_jmp_s_lr_decay
from mattertune.backbones import (
    MatterSimM3GNetBackboneModule,
    JMPBackboneModule,
    ORBBackboneModule,
    EqV2BackboneModule,
)

logging.basicConfig(level=logging.ERROR)



def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()
        
        if args_dict["model_type"] == "mattersim-1m":
            hparams.model = MC.MatterSimBackboneConfig.draft()
            hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
            hparams.model.pretrained_model = "MatterSim-v1.0.0-1M"
        elif args_dict["model_type"] == "mattersim-5m":
            hparams.model = MC.MatterSimBackboneConfig.draft()
            hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
            hparams.model.pretrained_model = "MatterSim-v1.0.0-5M"
        elif args_dict["model_type"] == "jmp-s":
            hparams.model = MC.JMPBackboneConfig.draft()
            hparams.model.graph_computer = MC.JMPGraphComputerConfig.draft()
            hparams.model.graph_computer.pbc = True
            hparams.model.pretrained_model = "jmp-s"
        elif args_dict["model_type"] == "orb-v2":
            hparams.model = MC.ORBBackboneConfig.draft()
            hparams.model.pretrained_model = "orb-v2"
        elif args_dict["model_type"] == "eqv2":
            hparams.model = MC.EqV2BackboneConfig.draft()
            hparams.model.checkpoint_path = Path(
                "/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/eqV2_31M_mp.pt"
            )
            hparams.model.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig.draft()
            hparams.model.atoms_to_graph.radius = 8.0
            hparams.model.atoms_to_graph.max_num_neighbors = 20
        else:
            raise ValueError(
                "Invalid model type, please choose from ['mattersim-1m', 'mattersim-5m', 'jmp-s', 'orb-v2', 'eqv2']"
            )
        hparams.model.reset_backbone = args_dict["reset_backbone"]
        hparams.model.reset_output_heads = True
        hparams.model.pruning_message_passing = args_dict["pruned_mp_steps"]
        hparams.model.optimizer = MC.AdamWConfig(
            lr=args_dict["lr"],
            amsgrad=False,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.1,
            per_parameter_hparams=get_jmp_s_lr_decay(args_dict["lr"]) if "jmp" in args_dict["model_type"] else None,
        )
        if args_dict["lr_scheduler"] == "steplr":
            hparams.model.lr_scheduler = MC.StepLRConfig(
                step_size=10, gamma=0.9
            )
        elif args_dict["lr_scheduler"] == "rlp":
            hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
                mode="min",
                monitor=f"val/forces_mae",
                factor=0.8,
                patience=5,
                min_lr=1e-8,
            )
        else:
            raise ValueError(
                "Invalid lr_scheduler, please choose from ['steplr', 'rlp']"
            )
            
        # Add model properties
        hparams.model.properties = []
        conservative = "mattersim" in args_dict["model_type"]
        energy = MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(), loss_coefficient=1.0
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(), conservative=conservative, loss_coefficient=1.0
        )
        hparams.model.properties.append(forces)

        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.train = MC.XYZDatasetConfig.draft()
        hparams.data.train.src = "/net/csefiles/coc-fung-cluster/lingyu/datasets/Pt_Ti_O_train.xyz"
        hparams.data.validation = MC.XYZDatasetConfig.draft()
        hparams.data.validation.src = "/net/csefiles/coc-fung-cluster/lingyu/datasets/Pt_Ti_O_val.xyz"
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
        hparams.trainer.max_epochs = args_dict["max_epochs"]
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.strategy = DDPStrategy(find_unused_parameters=True) if not "orb" in args_dict["model_type"] else DDPStrategy(static_graph=True, find_unused_parameters=True)
        hparams.trainer.gradient_clip_algorithm = "norm"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "32" if "mattersim" in args_dict["model_type"] else "bf16"

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/forces_mae", patience=args_dict["early_stop_patience"], mode="min", min_delta=1e-4
        )

        # Configure Model Checkpoint
        ckpt_name = f"{args_dict['model_type']}-best-MPx{args_dict['pruned_mp_steps']}"
        if args_dict["reset_backbone"]:
            ckpt_name += "-reset_backbone"
        if os.path.exists(f"./PtTiO-checkpoints/{ckpt_name}.ckpt"):
            os.remove(f"./PtTiO-checkpoints/{ckpt_name}.ckpt")
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_mae",
            dirpath="./PtTiO-checkpoints",
            filename=ckpt_name,
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Prune&Partition", 
                name=f"PtTiO-{args_dict['model_type']}-MPx{args_dict['pruned_mp_steps']}",
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

    ckpt_path = f"./PtTiO-checkpoints/{args_dict['model_type']}-best-MPx{args_dict['pruned_mp_steps']}"
    if args_dict["reset_backbone"]:
        ckpt_path += "-reset_backbone"
    ckpt_path += ".ckpt"
    
    if "mattersim" in args_dict["model_type"]:
        ft_model = MatterSimM3GNetBackboneModule.load_from_checkpoint(ckpt_path)
    elif "jmp" in args_dict["model_type"]:
        ft_model = JMPBackboneModule.load_from_checkpoint(ckpt_path)
    elif "orb" in args_dict["model_type"]:
        ft_model = ORBBackboneModule.load_from_checkpoint(ckpt_path)
    elif "eqv2" in args_dict["model_type"]:
        ft_model = EqV2BackboneModule.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(
            "Invalid model type, please choose from ['mattersim', 'jmp-s', 'orb-v2', 'eqv2']"
        )
    
    from ase.io import read
    from ase import Atoms
    import numpy as np
    import torch
    import wandb
    from tqdm import tqdm
    
    wandb.init(project="MatterTune-Prune&Partition", name=f"PtTiO-{args_dict['model_type']}-MPx{args_dict['pruned_mp_steps']}", resume=True)
    
    val_atoms_list:list[Atoms] = read("/net/csefiles/coc-fung-cluster/lingyu/datasets/Pt_Ti_O_val.xyz", ":") # type: ignore
    calc = ft_model.ase_calculator(
        device = f"cuda:{args_dict['devices'][0]}"
    )
    energies_per_atom = []
    forces = []
    pred_energies_per_atom = []
    pred_forces = []
    for atoms in tqdm(val_atoms_list):
        energies_per_atom.append(atoms.get_potential_energy() / len(atoms))
        forces.extend(np.array(atoms.get_forces()).tolist())
        atoms.set_calculator(calc)
        pred_energies_per_atom.append(atoms.get_potential_energy() / len(atoms))
        pred_forces.extend(np.array(atoms.get_forces()).tolist())
        
    e_mae = torch.nn.L1Loss()(torch.tensor(energies_per_atom), torch.tensor(pred_energies_per_atom))
    f_mae = torch.nn.L1Loss()(torch.tensor(forces), torch.tensor(pred_forces))
    
    rich.print(f"Energy MAE: {e_mae} eV/atom")
    rich.print(f"Forces MAE: {f_mae} eV/Ang")
    wandb.finish()
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mattersim-1m")
    parser.add_argument("--reset_backbone", action="store_true")
    parser.add_argument("--pruned_mp_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--lr_scheduler", type=str, default="rlp")
    parser.add_argument("--early_stop_patience", type=int, default=50)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
