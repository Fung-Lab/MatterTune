from __future__ import annotations

import rich
import os
from pathlib import Path
from ase import Atoms
from ase.io import read
import numpy as np
import torch
from lightning.pytorch.strategies import DDPStrategy

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig
from mattertune.main import load_finetuned_checkpoint

import matplotlib.pyplot as plt

EXAMPLE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(
    "/net/csefiles/coc-fung-cluster/lingyu/electrolyte/all-train-ase-eVA.xyz")
ENERGY_REFERENCE_PATH = EXAMPLE_DIR / "data" / \
    "Li-system-train-ase-energy_reference.json"
CHECKPOINT_DIR = EXAMPLE_DIR / "checkpoints"


def normalize_devices(raw_devices: list[int | str]) -> list[int]:
    devices: list[int] = []
    for item in raw_devices:
        if isinstance(item, int):
            devices.append(item)
            continue
        for piece in str(item).split(","):
            piece = piece.strip()
            if not piece:
                continue
            devices.append(int(piece))
    if not devices:
        raise ValueError("At least one device must be provided.")
    return devices


def infer_model_family(model_type: str) -> str:
    lowered = model_type.strip().lower()
    if lowered.startswith("mattersim"):
        return "mattersim"
    if lowered.startswith("uma"):
        return "uma"
    if lowered.startswith("orb"):
        return "orb"
    if lowered.startswith("mace"):
        return "mace"

    raise ValueError(
        f"Unsupported model type `{model_type}`. "
        "Expected a MatterSim, UMA, ORB, or MACE pretrained model name."
    )


def configure_model(hparams, args_dict: dict):
    model_type = args_dict["model_type"]
    model_family = infer_model_family(model_type)

    if model_family == "mattersim":
        hparams.model = MC.MatterSimBackboneConfig.draft()
        hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        hparams.model.pretrained_model = model_type
    elif model_family == "mace":
        hparams.model = MC.MACEBackboneConfig.draft()
        hparams.model.pretrained_model = model_type
    elif model_family == "uma":
        hparams.model = MC.UMABackboneConfig.draft()
        hparams.model.model_name = model_type
        hparams.model.task_name = args_dict["task_name"]
    elif model_family == "orb":
        hparams.model = MC.ORBBackboneConfig.draft()
        hparams.model.pretrained_model = model_type
        hparams.model.system = MC.ORBSystemConfig.draft()
        hparams.model.system.radius = args_dict["orb_radius"]
        hparams.model.system.max_num_neighbors = args_dict["orb_max_num_neighbors"]
    else:
        raise AssertionError(f"Unhandled model family: {model_family}")

    hparams.model.ignore_gpu_batch_transform_error = True
    hparams.model.freeze_backbone = False
    hparams.model.reset_output_heads = True
    return model_family


def build_config(args_dict: dict):
    hparams = MC.MatterTunerConfig.draft()
    model_family = configure_model(hparams, args_dict)
    hparams.model.optimizer = MC.AdamWConfig(
        lr=args_dict["lr"],
        amsgrad=False,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
        mode="min",
        monitor="val/total_loss",
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

    # Data Hyperparameters
    hparams.data = MC.AutoSplitDataModuleConfig.draft()
    hparams.data.dataset = MC.XYZDatasetConfig.draft()
    hparams.data.dataset.src = str(DATASET_PATH)
    hparams.data.train_split = 0.9
    hparams.data.shuffle = True
    hparams.data.shuffle_seed = 42
    hparams.data.batch_size = args_dict["batch_size"]
    hparams.data.pin_memory = False

    # Add Normalization for Energy
    hparams.model.normalizers = {
        "energy": [
            MC.PerAtomReferencingNormalizerConfig(
                per_atom_references=ENERGY_REFERENCE_PATH
            ),
            # MC.PerAtomNormalizerConfig(),
        ]
    }

    # Trainer Hyperparameters
    hparams.trainer = MC.TrainerConfig.draft()
    hparams.trainer.max_epochs = 10
    hparams.trainer.accelerator = "gpu"
    hparams.trainer.devices = args_dict["devices"]
    if len(args_dict["devices"]) > 1:
        hparams.trainer.strategy = DDPStrategy()
    hparams.trainer.gradient_clip_algorithm = "norm"
    hparams.trainer.gradient_clip_val = 1.0
    hparams.trainer.precision = "32"

    # Configure EMA
    hparams.trainer.ema = MC.EMAConfig(decay=0.99)

    # Configure Early Stopping
    hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
        monitor="val/total_loss", patience=50, mode="min", min_delta=1e-4
    )

    # Configure Model Checkpoint
    ckpt_name = args_dict["model_type"] + "-best"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{ckpt_name}.ckpt"
    if ckpt_path.exists():
        os.remove(ckpt_path)
    hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
        monitor="val/total_loss",
        dirpath=str(CHECKPOINT_DIR),
        filename=ckpt_name,
        save_top_k=1,
        mode="min",
    )

    # Configure Logger
    hparams.trainer.loggers = [
        WandbLoggerConfig(
            project="MatterTune-Electrolyte",
            name=f"{model_family}-{args_dict['model_type']}-Li-system",
        )
    ]

    # Additional trainer settings
    hparams.trainer.additional_trainer_kwargs = {
        "inference_mode": False,
    }

    return hparams.finalize(strict=False)


def main(args_dict: dict):
    mt_config = build_config(args_dict)
    model, trainer = MatterTuner(mt_config).tune()

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    model = load_finetuned_checkpoint(best_ckpt_path)
    calc = model.ase_calculator(
        device=f"cuda:{args_dict['devices'][0]}"
    )

    test_atoms_list: list[Atoms] = read(
        "/net/csefiles/coc-fung-cluster/lingyu/electrolyte/all-test-ase-eVA.xyz", ":")  # type: ignore

    energy_gt_list = []
    energy_pred_list = []
    forces_gt_list = []
    forces_pred_list = []
    natoms_list = []
    for atoms in test_atoms_list:
        natoms_list.append(len(atoms))
        energy_gt_list.append(atoms.get_potential_energy())
        forces_gt_list.append(atoms.get_forces())
        atoms.set_calculator(calc)
        energy_pred_list.append(atoms.get_potential_energy())
        forces_pred_list.append(atoms.get_forces())

    energy_gt_list = np.array(energy_gt_list)
    energy_pred_list = np.array(energy_pred_list)
    forces_gt_list = np.vstack(forces_gt_list)
    forces_pred_list = np.vstack(forces_pred_list)

    e_per_atom_gt_list = energy_gt_list / natoms_list
    e_per_atom_pred_list = energy_pred_list / natoms_list

    e_mae = torch.nn.L1Loss()(torch.tensor(e_per_atom_gt_list),
                              torch.tensor(e_per_atom_pred_list))
    f_mae = torch.nn.L1Loss()(torch.tensor(forces_gt_list),
                              torch.tensor(forces_pred_list))
    e_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(
        e_per_atom_gt_list), torch.tensor(e_per_atom_pred_list)))
    f_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(
        forces_gt_list), torch.tensor(forces_pred_list)))
    rich.print(f"Energy MAE: {e_mae} eV/atom")
    rich.print(f"Forces MAE: {f_mae} eV/Ang")
    rich.print(f"Energy RMSE: {e_rmse} eV/atom")
    rich.print(f"Forces RMSE: {f_rmse} eV/Ang")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(energy_gt_list, energy_pred_list)
    plt.plot(plt.xlim(), plt.ylim(), transform=plt.transAxes,
             linestyle="-", color="k", alpha=0.7)
    plt.xlabel("True Energy (eV)")
    plt.ylabel("Predicted Energy (eV)")
    plt.title("Potential Energy")
    plt.subplot(1, 2, 2)
    plt.scatter(forces_gt_list, forces_pred_list)
    plt.plot(plt.xlim(), plt.ylim(), transform=plt.transAxes,
             linestyle="-", color="k", alpha=0.7)
    plt.xlabel("True Forces (eV/Ang)")
    plt.ylabel("Predicted Forces (eV/Ang)")
    plt.title("Forces")
    plt.savefig("electrolyte_parity_plot.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str,
                        default="MatterSim-v1.0.0-1M")
    parser.add_argument("--task_name", type=str, default="omat")
    parser.add_argument("--orb_radius", type=float, default=6.0)
    parser.add_argument("--orb_max_num_neighbors", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2"])
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["devices"] = normalize_devices(args_dict["devices"])
    main(args_dict)
