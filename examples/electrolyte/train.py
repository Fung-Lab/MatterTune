from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import rich
import torch
from ase import Atoms
from ase.io import read
from lightning.pytorch.strategies import DDPStrategy

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig
from mattertune.main import load_finetuned_checkpoint

import matplotlib.pyplot as plt

EXAMPLE_DIR = Path(__file__).resolve().parent
ENERGY_REFERENCE_PATH = EXAMPLE_DIR / "data" / \
    "Li-system-train-ase-energy_reference.json"
CHECKPOINT_DIR = EXAMPLE_DIR / "checkpoints"


def _json_sanitize(obj: object) -> object:
    """Make a structure safe for wandb.init(config=...) JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _cli_args_for_wandb(args_dict: dict) -> dict:
    """CLI snapshot: paths as strings, devices already JSON-friendly."""
    return _json_sanitize(dict(args_dict))


def _mattertune_config_dict(hparams) -> dict:
    """Full MatterTune draft config as plain dict (matches the run after finalize)."""
    return json.loads(hparams.model_dump_json())


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
        monitor=args_dict["monitor"],
        factor=0.8,
        patience=5,
        min_lr=1e-8,
    )

    # Add model properties
    hparams.model.properties = []
    energy = MC.EnergyPropertyConfig(
        loss=MC.HuberLossConfig(delta=0.1), loss_coefficient=args_dict["e_loss_weight"]
    )
    hparams.model.properties.append(energy)
    forces = MC.ForcesPropertyConfig(
        loss=MC.HuberLossConfig(delta=0.1),
        loss_coefficient=args_dict["f_loss_weight"], conservative=True
    )
    hparams.model.properties.append(forces)

    # Data Hyperparameters
    hparams.data = MC.AutoSplitDataModuleConfig.draft()
    hparams.data.dataset = MC.XYZDatasetConfig.draft()
    hparams.data.dataset.src = args_dict["train_file"]
    hparams.data.train_split = args_dict["train_split"]
    hparams.data.shuffle = True
    hparams.data.shuffle_seed = 42
    hparams.data.batch_size = args_dict["batch_size"]
    hparams.data.pin_memory = False

    # Add Normalization for Energy
    energy_normalizer = [
        MC.PerAtomReferencingNormalizerConfig(
            per_atom_references=Path(args_dict["energy_reference"])
        ),
    ]
    if args_dict["per_atom_energy_normalize"]:
        energy_normalizer.append(MC.PerAtomNormalizerConfig())
    hparams.model.normalizers = {
        "energy": energy_normalizer
    }

    # Trainer Hyperparameters
    hparams.trainer = MC.TrainerConfig.draft()
    hparams.trainer.max_epochs = args_dict["max_epochs"]
    hparams.trainer.accelerator = args_dict["accelerator"]
    hparams.trainer.devices = args_dict["devices"]
    if len(args_dict["devices"]) > 1:
        hparams.trainer.strategy = DDPStrategy()
    hparams.trainer.gradient_clip_algorithm = "norm"
    hparams.trainer.gradient_clip_val = 2.0
    hparams.trainer.precision = "32"

    # Configure EMA
    hparams.trainer.ema = MC.EMAConfig(decay=0.99)

    # Configure Early Stopping
    hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
        monitor=args_dict["monitor"], patience=args_dict["patience"], mode="min", min_delta=1e-5
    )

    # Configure Model Checkpoint
    ckpt_name = args_dict["model_type"] + "-best"
    checkpoint_dir = Path(args_dict["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{ckpt_name}.ckpt"
    if ckpt_path.exists():
        os.remove(ckpt_path)
    hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
        monitor=args_dict["monitor"],
        dirpath=str(checkpoint_dir),
        filename=ckpt_name,
        save_top_k=1,
        mode="min",
    )

    log_dir = Path(args_dict["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    wandb_run_name = (args_dict.get("wandb_name") or "").strip()
    if not wandb_run_name:
        wandb_run_name = (
            f"{datetime.now().strftime('%Y%m%d-%H%M')}-{args_dict['model_type']}"
        )

    wandb_init_config = {
        "cli": _cli_args_for_wandb(args_dict),
        "mattertune": _mattertune_config_dict(hparams),
    }

    if args_dict["logger"] == "wandb":
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project=args_dict["wandb_project"],
                name=wandb_run_name,
                offline=False,
                save_dir=str(log_dir),
                additional_init_parameters={"config": wandb_init_config},
            )
        ]
    elif args_dict["logger"] == "csv":
        hparams.trainer.loggers = [
            MC.CSVLoggerConfig(save_dir=str(log_dir), name="lightning_logs")
        ]
    else:
        raise ValueError(
            f"Unsupported --logger {args_dict['logger']!r}; use 'wandb' or 'csv'."
        )

    # Additional trainer settings
    hparams.trainer.additional_trainer_kwargs = {
        "inference_mode": False,
    }

    return hparams.finalize(strict=False)


def main(args_dict: dict):
    mt_config = build_config(args_dict)
    model, trainer = MatterTuner(mt_config).tune()

    if args_dict.get("skip_eval"):
        rich.print("skip_eval set; skipping test-set parity plot and metrics.")
        return

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    model = load_finetuned_checkpoint(best_ckpt_path)
    eval_dev = (args_dict.get("eval_device") or "").strip()
    calc_device = eval_dev if eval_dev else f"cuda:{args_dict['devices'][0]}"
    calc = model.ase_calculator(device=calc_device)

    test_atoms_list: list[Atoms] = read(
        args_dict["test_file"], ":")  # type: ignore
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

    energy_gt = np.asarray(energy_gt_list).reshape(-1)
    energy_pred = np.asarray(energy_pred_list).reshape(-1)

    forces_gt = np.asarray(forces_gt_list).reshape(-1)
    forces_pred = np.asarray(forces_pred_list).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # -------------------------
    # Energy parity plot
    # -------------------------
    ax = axes[0]
    ax.scatter(energy_gt, energy_pred, s=10, alpha=0.6)

    emin = min(energy_gt.min(), energy_pred.min())
    emax = max(energy_gt.max(), energy_pred.max())
    ax.plot([emin, emax], [emin, emax], linestyle="-", color="k", alpha=0.7)

    ax.set_xlim(emin, emax)
    ax.set_ylim(emin, emax)
    ax.set_xlabel("True Energy (eV)")
    ax.set_ylabel("Predicted Energy (eV)")
    ax.set_title("Potential Energy")
    ax.set_aspect("equal", adjustable="box")

    # -------------------------
    # Force parity plot
    # -------------------------
    ax = axes[1]
    ax.scatter(forces_gt, forces_pred, s=2, alpha=0.3)

    fmin = min(forces_gt.min(), forces_pred.min())
    fmax = max(forces_gt.max(), forces_pred.max())
    ax.plot([fmin, fmax], [fmin, fmax], linestyle="-", color="k", alpha=0.7)

    ax.set_xlim(fmin, fmax)
    ax.set_ylim(fmin, fmax)
    ax.set_xlabel("True Forces (eV/Ang)")
    ax.set_ylabel("Predicted Forces (eV/Ang)")
    ax.set_title("Forces")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig("./parity_plot.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str,
                        default="MatterSim-v1.0.0-1M")
    parser.add_argument("--task_name", type=str,
                        default="omat")  # only for UMA
    parser.add_argument("--orb_radius", type=float,
                        default=6.0)  # only for ORB
    parser.add_argument("--orb_max_num_neighbors",
                        type=int, default=120)  # only for ORB
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--train_file", type=str,
                        default="/net/csefiles/coc-fung-cluster/lingyu/electrolyte/all-train-ase-eVA.xyz")
    parser.add_argument("--test_file", type=str,
                        default="/net/csefiles/coc-fung-cluster/lingyu/electrolyte/all-test-ase-eVA.xyz")
    parser.add_argument("--energy_reference", type=str,
                        default=str(ENERGY_REFERENCE_PATH))
    parser.add_argument("--checkpoint_dir", type=str,
                        default=str(CHECKPOINT_DIR))
    parser.add_argument("--log_dir", type=str,
                        default=str(EXAMPLE_DIR / "logs"))
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--logger", type=str, default="wandb",
                        choices=["wandb", "csv"])
    parser.add_argument("--wandb_project", type=str,
                        default="MatterTune-Electrolyte")
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="",
        help="W&B run name; empty -> YYYYMMDD-HHMM-<model_type>",
    )
    parser.add_argument("--eval_device", type=str, default="",
                        help="Device for post-train ASE eval; default first --devices GPU")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--per_atom_energy_normalize", action="store_true")
    parser.add_argument("--e_loss_weight", type=float, default=1.0)
    parser.add_argument("--f_loss_weight", type=float, default=1.0)
    parser.add_argument("--monitor", type=str, default="val/forces_mae")
    parser.add_argument("--patience", type=int, default=200)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["devices"] = normalize_devices(args_dict["devices"])
    main(args_dict)
