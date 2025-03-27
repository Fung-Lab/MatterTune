from __future__ import annotations

import copy
import logging
import os
from typing import cast

from ase import Atoms
from ase.constraints import UnitCellFilter
from ase.io import read, write
from ase.optimize import BFGS

from mattertune.backbones import JMPBackboneModule

logging.basicConfig(level=logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def main(args_dict: dict):
    ## Load Checkpoint and Create ASE Calculator
    model = JMPBackboneModule.load_from_checkpoint(
        checkpoint_path=args_dict["checkpoint_path"], map_location="cpu"
    )
    calc = model.ase_calculator(device=f"cuda:{args_dict['devices']}")

    ## Perform Structure Optimization with BFGS
    os.makedirs(args_dict["save_dir"], exist_ok=True)
    files = os.listdir(args_dict["init_structs"])
    for file in files:
        if os.path.exists(os.path.join(args_dict["save_dir"], file)):
            continue
        atoms = read(os.path.join(args_dict["init_structs"], file))
        assert isinstance(atoms, Atoms), "Expected an Atoms object"
        relax_traj = []
        idx = int(file.split(".")[0])
        atoms.pbc = True
        atoms.calc = calc
        energy_0 = atoms.get_potential_energy()
        ucf = UnitCellFilter(atoms, scalar_pressure=0.0)
        ucf_as_atoms = cast(Atoms, ucf)
        # UnitCellFilter is not a subclass of Atoms, but it can fill the role of Atoms in nearly all contexts
        opt = BFGS(ucf_as_atoms, logfile=None)

        def write_traj():
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
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    args_dict = vars(parser.parse_args())
    main(args_dict)
