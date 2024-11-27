from __future__ import annotations

import copy
import logging
import os

import ase.units as units
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin

from tqdm import tqdm

from mattertune.backbones import (
    JMPBackboneModule,
    M3GNetBackboneModule,
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
    else:
        raise ValueError(
            "Invalid fine-tuning model, must be one of 'jmp', 'orb', or 'm3gnet'."
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
    atoms: Atoms = read(args_dict["init_struct"])
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
        settings=wandb.Settings(code_dir=None, _disable_stats=True),
    )
    wandb.config.update(args_dict)

    ## Run Langevin Dynamics
    dyn = Langevin(
        atoms,
        temperature_K=args_dict["temperature"],
        timestep=args_dict["timestep"] * units.fs,
        friction=args_dict["friction"],
    )

    # Attach trajectory writing
    def attach_func():
        scaled_pos = dyn.atoms.get_scaled_positions()
        dyn.atoms.set_scaled_positions(np.mod(scaled_pos, 1))
        temp = dyn.atoms.get_temperature()
        e = dyn.atoms.get_potential_energy()
        f = dyn.atoms.get_forces()
        avg_f = np.mean(np.linalg.norm(f, axis=1))
        write(
            f"./md_results/md_traj_fric{args_dict['friction']}_{model_name}.xyz",
            copy.deepcopy(dyn.atoms),
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

    dyn.attach(attach_func, interval=args_dict["interval"])
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
        "--ckpt_path", type=str, default="./checkpoints/m3gnet-best.ckpt"
    )
    parser.add_argument("--init_struct", type=str, default="./data/H2O.xyz")
    parser.add_argument("--devices", type=int, nargs="+", default=[1])
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20000)
    args_dict = vars(parser.parse_args())
    main(args_dict)
