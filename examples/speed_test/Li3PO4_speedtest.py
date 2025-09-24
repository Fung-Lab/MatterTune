from __future__ import annotations

import logging
import os
import time
import rich
from tqdm import tqdm
import wandb

import numpy as np
from ase.io import read, write
import ase.units as units
from ase import Atoms
from ase.md.langevin import Langevin

from mattertune.backbones import (
    JMPBackboneModule,
    MatterSimM3GNetBackboneModule,
    ORBBackboneModule,
    MACEBackboneModule,
)
from mattertune.util import set_global_random_seed
from mattertune.wrappers.ase_calculator import MatterTunePartitionCalculator
from mattertune.wrappers.utils.parallel_inference import ParallizedInferenceDDP


logging.basicConfig(level=logging.ERROR)


def main(args_dict: dict):
    model_name = args_dict["ckpt_path"].split("/")[-1].replace(".ckpt", "")
    args_dict["mp_steps"] = int(model_name.split("-MPx")[-1][0])
    print(f"MP steps: {args_dict['mp_steps']}")
    assert f"MPx{args_dict['mp_steps']}" in model_name, "MP steps in checkpoint name does not match the provided mp_steps argument."
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
    elif "mace" in args_dict["ckpt_path"].lower():
        model = MACEBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    else:
        raise ValueError(
            "Invalid fine-tuning model, must be one of 'jmp', 'orb', 'mattersim'"
        )
        
    if args_dict["strategy"] == "single":
        calc = model.ase_calculator(device=f"cuda:{args_dict['devices'][0]}")
    elif args_dict["strategy"] == "ddp":
        inferencer = ParallizedInferenceDDP(
            ckpt_path=args_dict["ckpt_path"],
            properties=["energy", "forces"],
            devices=args_dict["devices"],
            batch_size=args_dict["batch_size"],
            num_workers=args_dict["num_workers"],
            using_partition=True,
            tmp_dir=args_dict["tmp_dir"],
        )
        calc = MatterTunePartitionCalculator(
            model=model,
            inferencer=inferencer,
            mp_steps=args_dict["mp_steps"],
            granularity=args_dict["granularity"],
        )
    else:
        raise ValueError(f"Invalid strategy: {args_dict['strategy']}. Must be 'single' or 'ddp'.")
    
    atoms:Atoms = read("./li3po4_quench_192.xyz") # type: ignore
    expected_size = args_dict["expected_size"]
    k = 2
    while True:
        k_atoms = atoms * (k, k, k)
        kminus1_atoms = atoms * (k-1, k-1, k-1)
        kplus1_atoms = atoms * (k+1, k+1, k+1)
        if abs(len(k_atoms) - expected_size) < abs(len(kminus1_atoms) - expected_size) and abs(len(k_atoms) - expected_size) < abs(len(kplus1_atoms) - expected_size):
            break
        k += 1
        if k > 10000:
            raise ValueError("Cannot find a suitable supercell size.")
    atoms = k_atoms
    system = f"Li3PO4-{len(atoms)}-{model_name}"
    atoms.pbc = True
    atoms.calc = calc
    
    wandb.init(
        project="Prune&Partition-SpeedTest",
        name=f"{system}-{args_dict['strategy']}",
        config=args_dict,
        save_code=False,
    )
    
    ## 600K Langevin dynamics
    num_steps = args_dict["num_steps"]
    # assert num_steps >= 50, "Number of steps must be at least 200 for Langevin dynamics."
    dyn = Langevin(
        atoms,
        temperature_K=600,
        timestep=1 * units.fs,
        friction=0.02,
    )
    md_step_times = []
    pbar = tqdm(range(num_steps), desc=f"Langevin at 600 K")
    for i in range(num_steps):
        start_time = time.time()
        dyn.step(1)
        end_time = time.time()
        if i > 20: # skip the first 20 steps for warm-up
            wandb.log(
                {
                    "md_step_time": end_time - start_time,
                    "partition_time": calc.partition_times[-1],
                    "forward_time": calc.forward_times[-1],
                    "collect_time": calc.collect_times[-1],
                }, 
                step=i
            )
            md_step_times.append(end_time - start_time)
        pbar.update(1)
    pbar.close()
    
    md_step_time = np.mean(md_step_times)
    ## speed in Microseconds/(atom ⋅ step)
    speed = md_step_time * 1e6 / len(atoms)
    rich.print(
        f"Average MD step time: {md_step_time:.4f} s, Speed: {speed:.2f} μs/(atom⋅step)"
    )
    
    wandb.finish()
    
    if args_dict["strategy"] == "ddp":
        inferencer.exit()

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speed test for MatterTune partition calculator.")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--expected_size", type=int, default=1000, help="Expected number of atoms in the input structure."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["single", "ddp"],
        default="ddp",
        help="Strategy for running the calculator, either 'single' or 'ddp'.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
    )
    parser.add_argument("--granularity", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=50, help="Number of MD steps to run.")
    args = parser.parse_args()
    
    args_dict = vars(args)
    args_dict["tmp_dir"] = os.path.join(os.getcwd(), "tmp")
    os.makedirs(args_dict["tmp_dir"], exist_ok=True)
    set_global_random_seed(42)
    rich.print(f"Running speed test with arguments: {args_dict}")
    main(args_dict)
    
    os.system("ps aux | grep python > log.txt")
    with open("log.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "MatterTune" in line and "a.py" not in line:
                tokens = line.split()
                if len(tokens) > 1:
                    job_id = tokens[1]
                    os.system(f"kill -9 {job_id}")