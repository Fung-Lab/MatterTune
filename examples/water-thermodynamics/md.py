from __future__ import annotations

import copy
import logging
import os

import ase.units as units
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm import tqdm

from mattertune.backbones import JMPBackboneModule, M3GNetBackboneModule

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


def main(args_dict: dict):
    ## Load Checkpoint and Create ASE Calculator
    model = M3GNetBackboneModule.load_from_checkpoint(
        checkpoint_path=args_dict["checkpoint_path"], map_location="cpu"
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
    if os.path.exists("./md_results/md_traj.xyz"):
        os.remove("./md_results/md_traj.xyz")

    ## Initialize WandB
    import wandb

    wandb.init(
        project="MatterTune-Examples",
        name="Water-NVT-{}".format(
            args_dict["checkpoint_path"].split("/")[-1].split(".")[0]
        ),
        save_code=False,
        settings=wandb.Settings(code_dir=None, _disable_stats=True),
    )
    wandb.config.update(args_dict)

    ## Run Langevin Dynamics
    md_traj = []
    # Langevin dynamics with new target temperature
    dyn = Langevin(
        atoms,
        temperature_K=args_dict["temperature"],
        timestep=args_dict["timestep"] * units.fs,
        friction=args_dict["friction"],
    )

    # Attach trajectory writing
    def store_traj():
        md_traj.append(copy.deepcopy(atoms))

    dyn.attach(store_traj, interval=args_dict["interval"])
    pbar = tqdm(
        range(args_dict["steps"]), desc=f"Langevin at {args_dict['temperature']} K"
    )
    for i in range(args_dict["steps"]):
        dyn.run(1)
        scaled_pos = atoms.get_scaled_positions()
        atoms.set_scaled_positions(np.mod(scaled_pos, 1))
        pbar.set_postfix(
            {
                "Energy": atoms.get_potential_energy(),
                "Temperature": atoms.get_temperature(),
            }
        )
        pbar.update(1)
        temp = atoms.get_temperature()
        e = atoms.get_potential_energy()
        wandb.log(
            {
                "Temperature (K)": temp,
                "Energy (eV)": e,
                "Time (fs)": i * args_dict["timestep"],
            }
        )
    pbar.close()

    ## Write trajectory to file
    write("./md_results/md_traj.xyz", md_traj)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, default="./checkpoints/m3gnet-best.ckpt"
    )
    parser.add_argument("--init_struct", type=str, default="./data/H2O.xyz")
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20000)
    args_dict = vars(parser.parse_args())
    main(args_dict)
