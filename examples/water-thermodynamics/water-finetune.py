from __future__ import annotations

import logging
from pathlib import Path
import rich

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
nu.pretty()



def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()
        
        if args_dict["model_type"] == "mattersim-1m":
            hparams.model = MC.MatterSimBackboneConfig.draft()
            hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
            hparams.model.pretrained_model = "MatterSim-v1.0.0-1M"
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
                "Invalid model type, please choose from ['mattersim-1m', 'jmp-s', 'orb-v2']"
            )
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.freeze_backbone = False
        hparams.model.reset_output_heads = True
        hparams.model.optimizer = MC.AdamWConfig(
            lr=8.0e-5,
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
        hparams.trainer.ema = MC.EMAConfig(decay=args_dict["ema_decay"])

        # Add model properties
        hparams.model.properties = []
        energy_coefficient = 1.0 / (192**2)
        conservative = args_dict["conservative"] or "mattersim" in args_dict["model_type"]
        energy = MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(), loss_coefficient=energy_coefficient
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(), conservative=conservative, loss_coefficient=1.0
        )
        hparams.model.properties.append(forces)

        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.train = MC.XYZDatasetConfig.draft()
        hparams.data.train.src = "./data/train_water_1000_eVAng.xyz"
        hparams.data.train.down_sample = args_dict["train_down_sample"]
        hparams.data.train.down_sample_refill = args_dict["down_sample_refill"]
        hparams.data.validation = MC.XYZDatasetConfig.draft()
        hparams.data.validation.src = "./data/val_water_1000_eVAng.xyz"
        hparams.data.batch_size = args_dict["batch_size"]

        ## Add Normalization for Energy
        hparams.model.normalizers = {
            "energy": [
                MC.PerAtomReferencingNormalizerConfig(
                    per_atom_references=Path(
                        "./data/water_1000_eVAng-energy_reference.json"
                    )
                )
            ]
        }

        ## Trainer Hyperparameters
        hparams.trainer = MC.TrainerConfig.draft()
        hparams.trainer.max_epochs = args_dict["max_epochs"]
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.strategy = DDPStrategy(find_unused_parameters=True) if not "orb" in args_dict["model_type"] else DDPStrategy(static_graph=True, find_unused_parameters=True)
        hparams.trainer.gradient_clip_algorithm = "norm"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "32"

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/forces_mae", patience=50, mode="min"
        )

        # Configure Model Checkpoint
        ckpt_name = f"{args_dict['model_type']}-best-{args_dict['train_down_sample']}"
        if args_dict["down_sample_refill"]:
            ckpt_name += "-refill"
        if args_dict["conservative"]:
            ckpt_name += "-conservative"
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_mae",
            dirpath="./checkpoints",
            filename=ckpt_name,
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Water-Finetune",
                name=f"Water-{args_dict['model_type']}-{args_dict['train_down_sample']}-refill_{args_dict['down_sample_refill']}",
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

    ckpt_path = f"./checkpoints/{args_dict['model_type']}-best-{args_dict['train_down_sample']}"
    if args_dict["down_sample_refill"]:
        ckpt_path += "-refill"
    if args_dict["conservative"]:
        ckpt_path += "-conservative"
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
            "Invalid model type, please choose from ['mattersim-1m', 'jmp-s', 'orb-v2', 'eqv2']"
        )
    
    from ase.io import read
    from ase import Atoms
    import numpy as np
    import torch
    import wandb
    
    wandb.init(project="MatterTune-Water-Finetune", name=f"Water-{args_dict['model_type']}-{args_dict['train_down_sample']}-refill_{args_dict['down_sample_refill']}", resume="allow")
    
    val_atoms_list:list[Atoms] = read("./data/val_water_1000_eVAng.xyz", ":") # type: ignore
    calc = ft_model.ase_calculator(
        device = f"cuda:{args_dict['devices'][0]}"
    )
    energies_per_atom = []
    forces = []
    pred_energies_per_atom = []
    pred_forces = []
    for atoms in val_atoms_list:
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
    parser.add_argument("--conservative", action="store_true")
    parser.add_argument("--train_down_sample", type=int, default=30)
    parser.add_argument("--down_sample_refill", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--lr_scheduler", type=str, default="steplr")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
