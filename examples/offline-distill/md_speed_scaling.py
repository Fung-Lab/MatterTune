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

from mattertune.students import CACEStudentModel

logging.basicConfig(level=logging.ERROR)

supercell_as = [1, 2, 3, 4, 5, 6, 7]


def main(args_dict: dict):
    
    md_atoms_base = read("./data/H2O.xyz")
    assert isinstance(md_atoms_base, Atoms), "Expected an Atoms object"
    
    
    val_atoms_list:list[Atoms] = read("./data/val_water_1593_eVAng.xyz", ":") # type: ignore
    model = CACEStudentModel.load_from_checkpoint("./checkpoints/cace-best.ckpt", lazy_init_atoms=val_atoms_list[0])
    calc = model.ase_calculator(device=f"cuda:{args_dict['device']}")
    
    atoms = read("./data/H2O.xyz")
    assert isinstance(atoms, Atoms), "Expected an Atoms object"
    atoms.pbc = True
    atoms.calc = calc

    ## Setup directories and remove old trajectory file
    save_dir = args_dict["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    ## Initialize WandB
    import wandb

    wandb.init(
        project="MatterTune-Examples",
        name="Distill-MD-Test-CACE",
        save_code=False,
    )

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
    if os.path.exists(os.path.join(save_dir, "water_cace_NPT.xyz")):
        os.remove(os.path.join(save_dir, "water_cace_NPT.xyz"))
        
    def log_func():
        temp = dyn.atoms.get_temperature()
        e = dyn.atoms.get_potential_energy()
        f = dyn.atoms.get_forces()
        avg_f = np.mean(np.linalg.norm(f, axis=1))
        write(
            os.path.join(save_dir, "water_cace_NPT.xyz"),
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
    parser.add_argument("--model", type=str, default="./checkpoints/cace-best.ckpt")
    parser.add_argument("--thermo_state", type=str, default="NPT")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=298)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--save_dir", type=str, default="/net/csefiles/coc-fung-cluster/lingyu/water_md")
    args_dict = vars(parser.parse_args())
    main(args_dict)
