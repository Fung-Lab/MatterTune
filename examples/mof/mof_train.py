from __future__ import annotations

import logging
from pathlib import Path
import rich
import os
import torch
import torch.distributed as dist

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
    MACEBackboneModule,
    UMABackboneModule,
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
        elif "orb" in args_dict["model_type"]:
            hparams.model = MC.ORBBackboneConfig.draft()
            hparams.model.pretrained_model = args_dict["model_type"]
        elif args_dict["model_type"] == "eqv2":
            hparams.model = MC.EqV2BackboneConfig.draft()
            hparams.model.checkpoint_path = Path(
                "/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/eqV2_31M_mp.pt"
            )
            hparams.model.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig.draft()
            hparams.model.atoms_to_graph.radius = 8.0
            hparams.model.atoms_to_graph.max_num_neighbors = 20
        elif "mace" in args_dict["model_type"]:
            hparams.model = MC.MACEBackboneConfig.draft()
            hparams.model.pretrained_model = args_dict["model_type"]
        elif "uma" in args_dict["model_type"]:
            hparams.model = MC.UMABackboneConfig.draft()
            hparams.model.model_name = args_dict["model_type"]
            hparams.model.task_name = "odac"
        else:
            raise ValueError(
                "Invalid model type, please choose from ['mattersim-1m', 'jmp-s', 'orb-v2']"
            )
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.freeze_backbone = False
        hparams.model.reset_backbone = False
        hparams.model.reset_output_heads = True
        hparams.model.optimizer = MC.AdamWConfig(
            lr=8.0e-5,
            amsgrad=False,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.1,
            per_parameter_hparams=get_jmp_s_lr_decay(args_dict["lr"]) if "jmp" in args_dict["model_type"] else None,
        )
        hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
            mode="min",
            monitor=f"val/forces_mae",
            factor=0.8,
            patience=5,
            min_lr=1e-8,
        )
        hparams.trainer.ema = MC.EMAConfig(decay=args_dict["ema_decay"])

        # Add model properties
        hparams.model.properties = []
        energy_coefficient = 1.0 
        conservative = args_dict["conservative"]
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
        hparams.data.train.src = "/nethome/lkong88/mof-foundational/databases/mof_train.xyz"
        hparams.data.validation = MC.XYZDatasetConfig.draft()
        hparams.data.validation.src = "/nethome/lkong88/mof-foundational/databases/mof_val.xyz"
        hparams.data.batch_size = args_dict["batch_size"]

        ## Add Normalization for Energy
        hparams.model.normalizers = {
            "energy": [
                MC.PerAtomReferencingNormalizerConfig(
                    per_atom_references=Path("./data/mof_train-energy_reference.json")
                ),
                MC.PerAtomNormalizerConfig(),
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
            monitor=f"val/forces_mae", patience=200, mode="min"
        )

        # Configure Model Checkpoint
        ckpt_name = f"{args_dict['model_type']}-best"
        if args_dict["conservative"]:
            ckpt_name += "-conservative"
        if os.path.exists(f"./checkpoints/{ckpt_name}.ckpt"):
            os.remove(f"./checkpoints/{ckpt_name}.ckpt")
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
                project="MatterTune-MOF-Finetune",
                name=f"MOF-{args_dict['model_type']}",
            )
        ]

        # Additional trainer settings
        hparams.trainer.additional_trainer_kwargs = {
            "inference_mode": False,
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    # mt_config = hparams()
    # model, trainer = MatterTuner(mt_config).tune()
    
    
    ## Perform Evaluation

    ckpt_path = f"./checkpoints/{args_dict['model_type']}-best"
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
    elif "mace" in args_dict["model_type"]:
        ft_model = MACEBackboneModule.load_from_checkpoint(ckpt_path)
    elif "uma" in args_dict["model_type"]:
        ft_model = UMABackboneModule.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(
            "Invalid model type, please choose from ['mattersim-1m', 'jmp-s', 'orb-v2', 'eqv2']"
        )
    
    import wandb
    from ase import Atoms
    from ase.io import read
    from mattertune.wrappers.ase_calculator import quick_efs_evaluation
    from mattertune.util import is_main_process
    
    if is_main_process():
        wandb.init(project="MatterTune-MOF-Finetune", name=f"MOF-{args_dict['model_type']}", resume="allow")
        torch.cuda.empty_cache()
        
        val_atoms_list: list[Atoms] = read("/nethome/lkong88/mof-foundational/databases/mof_val.xyz", index=":") # type: ignore

        results,_ = quick_efs_evaluation(
            model=ft_model,
            atoms_list=val_atoms_list,
            include_forces=True,
            include_stresses=False,
            device=f"cuda:{args_dict['devices'][0]}",
            metrics=["mae", "rmse"],
        )
        rich.print("Validation Set Results")
        rich.print(results)
        
        val_perturbation_atoms_list: list[Atoms] = read("/nethome/lkong88/mof-foundational/databases/mof_val_perturbations.xyz", index=":") # type: ignore
        results,_ = quick_efs_evaluation(
            model=ft_model,
            atoms_list=val_perturbation_atoms_list,
            include_forces=True,
            include_stresses=False,
            device=f"cuda:{args_dict['devices'][0]}",
            metrics=["mae", "rmse"],
        )
        rich.print("Validation with Perturbations Set Results")
        rich.print(results)
        
        val_solvation_atoms_list: list[Atoms] = read("/nethome/lkong88/mof-foundational/databases/mof_val_solvation.xyz", index=":") # type: ignore
        results,_ = quick_efs_evaluation(
            model=ft_model,
            atoms_list=val_solvation_atoms_list,
            include_forces=True,
            include_stresses=False,
            device=f"cuda:{args_dict['devices'][0]}",
            metrics=["mae", "rmse"],
        )
        rich.print("Validation with Solvation Set Results")
        rich.print(results)
        
        wandb.finish()
    
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mattersim-1m")
    parser.add_argument("--conservative", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    # parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--lr_scheduler", type=str, default="rlp")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
