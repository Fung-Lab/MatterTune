from __future__ import annotations

import logging
import os

from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from tqdm import tqdm

from mattertune.backbones import JMPBackboneModule

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


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
    os.makedirs("./relax_results", exist_ok=True)
    atoms_list: list[Atoms] = read(args_dict["init_structs"], index=":")
    relaxed_atoms_list = []
    pbar = tqdm(atoms_list, desc="Optimizing Structures")
    for i, atoms in enumerate(pbar):
        atoms.pbc = True
        atoms.calc = calc
        energy_0 = atoms.get_potential_energy()
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.01)
        energy_1 = atoms.get_potential_energy()
        relaxed_atoms_list.append(atoms)
        pbar.set_postfix({"Energy_0": energy_0, "Energy_1": energy_1})
    pbar.close()
    write("./relax_results/ZnMnO4_relaxed.xyz", relaxed_atoms_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, default="./checkpoints/jmp-best.ckpt"
    )
    parser.add_argument("--init_structs", type=str, default="./data/ZnMnO4_random.xyz")
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    args_dict = vars(parser.parse_args())
    main(args_dict)
