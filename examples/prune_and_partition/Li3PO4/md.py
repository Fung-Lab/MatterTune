from __future__ import annotations

import copy
import logging
import os
import time
import rich

import nshutils as nu
import ase.units as units
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm import tqdm
import torch.distributed as dist
import wandb
from datetime import datetime

from mattertune.backbones import (
    JMPBackboneModule,
    MatterSimM3GNetBackboneModule,
    ORBBackboneModule,
)
from mattertune.util import set_global_random_seed
from mattertune.wrappers.ase_calculator import MatterTunePartitionCalculator
from mattertune.wrappers.utils.nosehoover import NoseHoover
from mattertune.wrappers.utils.parallel_inference import (
    ParallizedInferenceDDP
)

logging.basicConfig(level=logging.ERROR)
nu.pretty()


def main(args_dict: dict):
    ## Load Checkpoint and Create ASE Calculator
    model_name = args_dict["ckpt_path"].split("/")[-1].replace(".ckpt", "")
    if "jmp" in args_dict["ckpt_path"]:
        model = JMPBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    elif "orb" in args_dict["ckpt_path"]:
        model = ORBBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    elif "mattersim" in args_dict["ckpt_path"].lower():
        model = MatterSimM3GNetBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    else:
        raise ValueError(
            "Invalid fine-tuning model, must be one of 'jmp', 'orb', 'mattersim'"
        )
    
    inferencer = ParallizedInferenceDDP(
        ckpt_path=args_dict["ckpt_path"],
        properties=["energy", "forces"],
        devices=args_dict["devices"],
        batch_size=args_dict["batch_size"],
        num_workers=args_dict["num_workers"],
        using_partition=True,
    )
    
    calc = MatterTunePartitionCalculator(
        model=model,
        inferencer=inferencer,
        mp_steps=args_dict["mp_steps"],
        granularity=args_dict["granularity"],
    )
    
    
    atoms = read(args_dict["init_struct"])
    system = args_dict["init_struct"].split("/")[-1].split(".")[0]
    assert isinstance(atoms, Atoms), "Expected an Atoms object"
    atoms.pbc = True
    atoms.calc = calc
    
    ## Run Langevin Dynamics
    if args_dict["thermo_state"].lower() == "langevin":
        dyn = Langevin(
            atoms,
            temperature_K=args_dict["temperature"],
            timestep=args_dict["timestep"] * units.fs,
            friction=args_dict["friction"],
        )
    elif args_dict["thermo_state"].lower() == "nosehoover":
        dyn = NoseHoover(
            atoms,
            temperature=args_dict["temperature"] * units.kB,
            timestep=args_dict["timestep"] * units.fs,
            nvt_q=5e5,
        )
    elif args_dict["thermo_state"].lower() == "npt":
        dyn = NPT(
            atoms,
            temperature_K=args_dict["temperature"],
            timestep=args_dict["timestep"] * units.fs,
            externalstress=None,
            ttime=100 * units.fs,
            pfactor=None,
        )
    else:
        raise ValueError("Invalid thermo_state, must be one of 'NVT' or 'NPT'")

    wandb.init(
        project="MatterTune-MD-with-Partition",
        name=f"{system}_{model_name}_{args_dict['thermo_state']}-{args_dict['temperature']}K-{args_dict['timestep']*args_dict['md_steps']/1000}ps",
        save_code=False,
        config=args_dict,
    )
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d %H:%M")
    save_dir = os.path.join(args_dict["save_dir"], f"{system}_{formatted_time}_{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    pbar = tqdm(range(args_dict["md_steps"]), desc=f"Langevin at {args_dict['temperature']} K")
    md_step_times = []
    md_overall_times = []
    for i in range(args_dict["md_steps"]):
        time1 = time.time()
        dyn.step()
        md_step_times.append(time.time() - time1)
        pbar.update(1)
        temp = dyn.atoms.get_temperature()
        pbar.set_postfix({"Temp": temp, "EnvTemp": args_dict["temperature"]})
        if i % args_dict["log_interval"] == 0:
            wandb.log(
                {
                    "Temperature (K)": temp,
                    "Time (fs)": i * args_dict["timestep"],
                    "MD Step Time (s)": md_step_times[-1],
                    "MD Overall Time (s)": time.time() - time1,
                },
                step=i,
            )
        if i % args_dict["save_interval"] == 0:
            current_time = i * args_dict["timestep"] # in fs
            pos = np.array(dyn.atoms.get_positions())
            atomic_numbers = np.array(dyn.atoms.get_atomic_numbers())
            cell = np.array(dyn.atoms.get_cell(complete=True))
            np.savez(
                os.path.join(save_dir, f"md_step_{i}.npz"),
                pos=pos,
                atomic_numbers=atomic_numbers,
                cell=cell,
            )
        md_overall_times.append(time.time() - time1)
    pbar.close()    
    average_md_step_time = np.mean(md_step_times)
    average_md_overall_times = np.mean(md_overall_times)
    rich.print(f"Storage Dir: {save_dir}")
    rich.print(f"Average MD Step Time: {average_md_step_time:.4f} seconds")
    rich.print(f"Average MD Overall Time: {average_md_overall_times:.4f} seconds")
    rich.print(f"Max MD Overall Time: {max(md_overall_times):.4f} seconds")
    rich.print(f"Min MD Overall Time: {min(md_overall_times):.4f} seconds")


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./Li3PO4-checkpoints/mattersim-1m-best-MPx1-first_half_0.02.ckpt",
    )
    parser.add_argument("--mp_steps", type=int, default=1)
    parser.add_argument("--granularity", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--show_inference_log", action="store_true")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--thermo_state", type=str, default="langevin")
    parser.add_argument("--init_struct", type=str, default="./init_structs/li3po4_quench_421824.xyz")
    parser.add_argument("--temperature", type=float, default=600)
    parser.add_argument("--timestep", type=float, default=2)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="/storage/lingyu/md_with_partition/")
    parser.add_argument("--tmp_dir", type=str, default="/storage/lingyu/tmp/")
    parser.add_argument("--md_steps", type=int, default=25000)
    args_dict = vars(parser.parse_args())
    
    set_global_random_seed(42)  # Set a global random seed for reproducibility    

    main(args_dict)
