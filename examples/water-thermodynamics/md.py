from __future__ import annotations

import copy
import logging
import os

import ase.units as units
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from tqdm import tqdm

from mattertune.backbones import (
    JMPBackboneModule,
    M3GNetBackboneModule,
    MatterSimM3GNetBackboneModule,
    ORBBackboneModule,
)

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


def main(args_dict: dict):
    ## Load Checkpoint and Create ASE Calculator
    model_name = args_dict["ckpt_path"].split("/")[-1].replace(".ckpt", "")
    if "jmp" in args_dict["ckpt_path"]:
        model_type = "jmp"
        model = JMPBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    elif "orb" in args_dict["ckpt_path"]:
        model_type = "orb"
        model = ORBBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    elif "m3gnet" in args_dict["ckpt_path"]:
        model_type = "m3gnet"
        model = M3GNetBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    elif "mattersim" in args_dict["ckpt_path"].lower():
        model_type = "mattersim"
        model = MatterSimM3GNetBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    else:
        raise ValueError(
            "Invalid fine-tuning model, must be one of 'jmp', 'orb', 'm3gnet', or 'mattersim'"
        )
    calc = model.ase_calculator(
        lightning_trainer_kwargs={
            "accelerator": "gpu",
            "devices": args_dict["devices"],
            "precision": "32",
            "inference_mode": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "barebones": True,
        }
    )
    atoms = read(args_dict["init_struct"])
    assert isinstance(atoms, Atoms), "Expected an Atoms object"
    atoms.pbc = True
    atoms.calc = calc

    ## Setup directories and remove old trajectory file
    os.makedirs("./md_results", exist_ok=True)

    ## Initialize WandB
    import wandb

    wandb.init(
        project="MatterTune-Examples",
        name="Water-NVT-{}".format(model_name),
        save_code=False,
    )
    wandb.config.update(args_dict)

    ## Run Langevin Dynamics
    if args_dict["thermo_state"].lower() == "nvt":
        dyn = Langevin(
            atoms,
            temperature_K=args_dict["temperature"],
            timestep=args_dict["timestep"] * units.fs,
            friction=args_dict["friction"],
        )
    elif args_dict["thermo_state"].lower() == "npt":
        dyn = NPT(
            atoms,
            temperature_K=args_dict["temperature"],  # 300 K
            timestep=args_dict["timestep"] * units.fs,  # 0.5 fs
            externalstress=None,
            ttime=100 * units.fs,
            pfactor=None,
        )
    else:
        raise ValueError("Invalid thermo_state, must be one of 'NVT' or 'NPT'")

    # Attach trajectory writing
    def log_func():
        temp = dyn.atoms.get_temperature()
        e = dyn.atoms.get_potential_energy()
        f = dyn.atoms.get_forces()
        avg_f = np.mean(np.linalg.norm(f, axis=1))
        write(
            os.path.join(
                "./md_results",
                f"water_{model_name}_{args_dict['thermo_state']}.xyz",
            ),
            dyn.atoms,
            append=True,
        )
        wandb.log(
            {
                "Temperature (K)": temp,
                "Energy (eV)": e,
                "Avg. Force (eV/Ang)": avg_f,
                "Time (fs)": i * args_dict["timestep"],
            }
        )

    dyn.attach(log_func, interval=args_dict["interval"])
    pbar = tqdm(
        range(args_dict["steps"]), desc=f"Langevin at {args_dict['temperature']} K"
    )
    for i in range(args_dict["steps"]):
        dyn.run(1)
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/MatterSim-v1.0.0-1M-bestTrue0.9.ckpt",
    )
    parser.add_argument("--thermo_state", type=str, default="NPT")
    parser.add_argument("--init_struct", type=str, default="./data/H2O.xyz")
    parser.add_argument("--devices", type=int, nargs="+", default=[3])
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400000)
    args_dict = vars(parser.parse_args())
    main(args_dict)
