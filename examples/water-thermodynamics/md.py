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
from mattertune.backbones import JMPBackboneModule, M3GNetBackboneModule
from tqdm import tqdm

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

    # Define temperature stages for gradual heating
    temperature_stages = args_dict["temperature_stages"]
    md_traj = []

    ## MD with multiple temperature stages
    for stage_idx, target_temp in enumerate(temperature_stages):
        print(f"Starting MD at {target_temp} K")

        # Re-initialize Langevin dynamics with new target temperature
        dyn = Langevin(
            atoms,
            temperature_K=target_temp,
            timestep=args_dict["timestep"] * units.fs,
            friction=args_dict["friction"],
        )

        # Reinitialize velocities for each stage
        # MaxwellBoltzmannDistribution(atoms, target_temp)

        # Attach trajectory writing
        def store_traj():
            md_traj.append(copy.deepcopy(atoms))

        dyn.attach(store_traj, interval=args_dict["interval"])

        # Run MD for this temperature stage
        steps_per_stage = args_dict["steps"][stage_idx]
        pbar = tqdm(range(steps_per_stage), desc=f"MD Stage {stage_idx+1}")
        for i in range(steps_per_stage):
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
                    "Temperature": temp,
                    "Energy": e,
                    "Time": (stage_idx * steps_per_stage + i) * args_dict["timestep"],
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
    parser.add_argument(
        "--temperature_stages", type=float, nargs="+", default=[100, 200, 300]
    )
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--steps", type=int, nargs="+", default=[20000, 20000, 20000])
    args_dict = vars(parser.parse_args())
    main(args_dict)
