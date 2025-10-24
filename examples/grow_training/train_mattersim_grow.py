from __future__ import annotations

import logging
from pathlib import Path
import rich
import os

from lightning.pytorch.strategies import DDPStrategy

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig
from mattertune.backbones import (
    MatterSimM3GNetBackboneModule,
    JMPBackboneModule,
    ORBBackboneModule,
    EqV2BackboneModule,
    MACEBackboneModule,
)

from mattertune.callbacks import PerLayerTrainDynamicsConfig

logging.basicConfig(level=logging.ERROR)



def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()
        if args_dict["model_type"] == "mattersim-1m":
            if args_dict["load_ckpt"] is not None:
                hparams.model = MC.MatterSimBackboneConfig.from_checkpoint(args_dict["load_ckpt"])
            else:
                hparams.model = MC.MatterSimBackboneConfig.draft()
                hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
                hparams.model.pretrained_model = "MatterSim-v1.0.0-1M"
        elif args_dict["model_type"] == "mattersim-5m":
            if args_dict["load_ckpt"] is not None:
                hparams.model = MC.MatterSimBackboneConfig.from_checkpoint(args_dict["load_ckpt"])
            else:
                hparams.model = MC.MatterSimBackboneConfig.draft()
                hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
                hparams.model.pretrained_model = "MatterSim-v1.0.0-5M"
        else:
            raise ValueError(
                "Invalid model type, please choose from ['mattersim-1m', 'mattersim-5m']"
            )
        hparams.model.reset_backbone = True
        hparams.model.reset_output_heads = True
        hparams.model.pruning_message_passing = args_dict["pruned_mp_steps"]
        hparams.model.freeze_group_bys = [f"backbone.model.graph_conv.{i}" for i in range(args_dict["pruned_mp_steps"]-1)]
        if len(hparams.model.freeze_group_bys) > 0:
            hparams.model.freeze_group_bys.append("backbone.model.edge_encoder")
            hparams.model.freeze_group_bys.append("backbone.model.atom_embedding")
        rich.print("Frozen layers:", hparams.model.freeze_group_bys)
        
        lr = args_dict["lr"]
        per_parameter_hparams = [
            {
                "patterns": ["model.atom_embedding.*"],
                "hparams": {
                    "lr": 0.3 * lr,
                },
            },
            {
                "patterns": ["model.edge_encoder.*"],
                "hparams": {
                    "lr": 0.55 * lr,
                },
            },
            {
                "patterns": ["model.graph_conv.0.*"],
                "hparams": {
                    "lr": 0.4 * lr,
                },
            },
            {
                "patterns": ["model.graph_conv.1.*"],
                "hparams": {
                    "lr": 0.3 * lr,
                },
            },
            {
                "patterns": ["model.graph_conv.2.*"],
                "hparams": {
                    "lr": 0.4 * lr,
                },
            },
            {
                "patterns": ["model.final.*"],
                "hparams": {
                    "lr": 0.55 * lr,
                },
            },
        ]
        
        hparams.model.optimizer = MC.AdamWConfig(
            lr=lr,
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

        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.train = MC.XYZDatasetConfig.draft()
        hparams.data.train.src = "/net/csefiles/coc-fung-cluster/lingyu/datasets/li3po4-train.xyz"
        hparams.data.validation = MC.XYZDatasetConfig.draft()
        hparams.data.validation.src = "/net/csefiles/coc-fung-cluster/lingyu/datasets/li3po4-val.xyz"
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
        hparams.trainer.precision = "32"
        hparams.trainer.log_every_n_steps = 1

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/forces_mae", patience=args_dict["early_stop_patience"], mode="min", min_delta=1e-4
        )

        # Configure Model Checkpoint
        ckpt_name = f"{args_dict['model_type']}-best-MPx{args_dict['pruned_mp_steps']}"
        if os.path.exists(f"./Li3PO4-checkpoints/{ckpt_name}.ckpt"):
            os.remove(f"./Li3PO4-checkpoints/{ckpt_name}.ckpt")
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/forces_mae",
            dirpath="./Li3PO4-checkpoints",
            filename=ckpt_name,
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-GrowTrain", 
                name=f"Li3PO4-{args_dict['model_type']}-MPx{args_dict['pruned_mp_steps']}",
            )
        ]
        
        # # Configure Per-Layer Training Dynamics Callback
        # hparams.trainer.additional_callbacks = [
        #     PerLayerTrainDynamicsConfig(
        #         log_every_n_steps = 1,
        #         group_by=r"^backbone\.model\.graph_conv\.\d+",
        #     )
        # ]

        # Additional trainer settings
        hparams.trainer.additional_trainer_kwargs = {
            "inference_mode": False,
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    model, trainer = MatterTuner(mt_config).tune()
    
    
    ## Perform Evaluation

    ckpt_path = f"./Li3PO4-checkpoints/{args_dict['model_type']}-best-MPx{args_dict['pruned_mp_steps']}"

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
    
    wandb.init(project="MatterTune-GrowTrain", name=f"Li3PO4-{args_dict['model_type']}-MPx{args_dict['pruned_mp_steps']}", resume=True)
    
    val_atoms_list:list[Atoms] = read("/net/csefiles/coc-fung-cluster/lingyu/datasets/li3po4-test.xyz", ":") # type: ignore
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
    e_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(energies_per_atom), torch.tensor(pred_energies_per_atom)))
    f_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(forces), torch.tensor(pred_forces)))
    
    rich.print(f"Energy MAE: {e_mae} eV/atom")
    rich.print(f"Forces MAE: {f_mae} eV/Ang")
    rich.print(f"Energy RMSE: {e_rmse} eV/atom")
    rich.print(f"Forces RMSE: {f_rmse} eV/Ang")
    wandb.finish()
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mattersim-1m")
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--down_sample", type=str, default=None)
    parser.add_argument("--pruned_mp_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--lr_scheduler", type=str, default="rlp")
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--conservative", action="store_true")
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
