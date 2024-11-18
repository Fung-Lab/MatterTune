from __future__ import annotations

import copy
import logging
import os

import numpy as np
from ase import Atoms
from ase.constraints import UnitCellFilter
from ase.io import read, write
from ase.optimize import LBFGS
from tqdm import tqdm

from mattertune.backbones import JMPBackboneModule

logging.basicConfig(level=logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def main(args_dict: dict):
    ## Load Checkpoint and Create ASE Calculator
    model = JMPBackboneModule.load_from_checkpoint(
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
        }
    )

    ## Perform Structure Optimization with BFGS
    os.makedirs(args_dict["save_dir"], exist_ok=True)
    files = os.listdir(args_dict["init_structs"])
    for file in files:
        atoms: Atoms = read(os.path.join(args_dict["init_structs"], file))
        relax_traj = []
        idx = int(file.split(".")[0])
        atoms.pbc = True
        atoms.calc = calc
        energy_0 = atoms.get_potential_energy()
        ucf = UnitCellFilter(atoms, scalar_pressure=0.0)
        opt = LBFGS(
            ucf,
            logfile=None,
        )
        pbar = tqdm(total=args_dict["max_steps"], desc=f"Structure {idx}")
        for step in range(args_dict["max_steps"]):
            opt.run(fmax=0.01, steps=1)
            energy_1 = atoms.get_potential_energy()
            pbar.set_postfix(
                {"Energy_0": f"{energy_0:.2f}", "Energy_1": f"{energy_1:.2f}"}
            )
            relax_traj.append(copy.deepcopy(atoms))
            pbar.update(1)
        pbar.close()
        write(
            os.path.join(args_dict["save_dir"], f"{idx}.xyz"),
            relax_traj,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, default="./checkpoints/jmp-best.ckpt"
    )
    parser.add_argument("--init_structs", type=str, default="./ZnMn2O4_random")
    parser.add_argument("--save_dir", type=str, default="./ZnMn2O4_mlrelaxed")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--devices", type=int, nargs="+", default=[2])
    args_dict = vars(parser.parse_args())
    main(args_dict)
