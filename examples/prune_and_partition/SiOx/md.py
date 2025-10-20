from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from tqdm import tqdm
import wandb

import numpy as np
import ase.units as units
from ase.io import read
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.npt import NPT

from mattertune.backbones import (
    JMPBackboneModule,
    MatterSimM3GNetBackboneModule,
    ORBBackboneModule,
)
from mattertune.util import set_global_random_seed
from mattertune.wrappers.ase_calculator import MatterTunePartitionCalculator
from mattertune.wrappers.utils.parallel_inference import ParallizedInferenceDDP


logging.basicConfig(level=logging.ERROR)

def get_pressure(stress: np.ndarray) -> float:
    """
    Calculate the pressure from the stress tensor.
    The stress tensor is expected to be in the format:
    [[s_xx, s_xy, s_xz],
     [s_yx, s_yy, s_yz],
     [s_zx, s_zy, s_zz]]
    """
    # Pressure is defined as -1/3 * trace(stress)
    return -np.trace(stress) / 3.0

def run_langevin(
    atoms: Atoms,
    calc: MatterTunePartitionCalculator,
    temperature: float,
    timestep: float,
    steps: int,
    friction: float,
    log_interval: int = 10,
    save_interval: int = 100,
    save_dir: str = "./",
    step_bias: int = 0,
) -> Atoms:
    """
    Run Langevin dynamics on the given atoms.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    dyn = Langevin(
        atoms,
        temperature_K=temperature,
        timestep=timestep * units.fs,
        friction=friction,
    )
    # Run Langevin dynamics
    md_step_times = []
    pbar = tqdm(range(steps), desc=f"Langevin at {temperature} K")
    for i in range(steps):
        time1 = time.time()
        dyn.step()
        md_step_times.append(time.time() - time1)
        pbar.update(1)
        temp = dyn.atoms.get_temperature()
        pbar.set_postfix({"Temp": temp, "EnvTemp": temperature})
        if i % log_interval == 0:
            partition_time = calc.partition_times[-1]
            forward_time = calc.forward_times[-1]
            collect_time = calc.collect_times[-1]
            partition_size = calc.partition_sizes[-1]
            wandb.log(
                {
                    "Temperature (K)": temp,
                    "Time (fs)": (i+step_bias) * timestep,
                    "Partition Time (s)": partition_time,
                    "Forward Time (s)": forward_time,
                    "Collect Time (s)": collect_time,
                    "Partition Size (atoms)": partition_size,
                    "MD Step Time (s)": md_step_times[-1],
                    "MD Overall Time (s)": time.time() - time1,
                },
                step=i+step_bias,
            )
        if i % save_interval == 0:
            pos = np.array(dyn.atoms.get_positions())
            atomic_numbers = np.array(dyn.atoms.get_atomic_numbers())
            cell = np.array(dyn.atoms.get_cell(complete=True))
            np.savez(
                os.path.join(save_dir, f"md_step_{i+step_bias}.npz"),
                pos=pos,
                atomic_numbers=atomic_numbers,
                cell=cell,
            )
    pbar.close()
    return atoms

def run_npt(
    atoms: Atoms,
    calc: MatterTunePartitionCalculator,
    temperature: float,
    timestep: float,
    steps: int,
    externalstress: float | None = None,
    ttime: float | None = None,
    pfactor: float | None = None,
    log_interval: int = 10,
    save_interval: int = 100,
    save_dir: str = "./",
    step_bias: int = 0,
) -> Atoms:
    """
    Run NPT dynamics on the given atoms.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    dyn = NPT(
        atoms,
        temperature_K=temperature,
        timestep=timestep * units.fs,
        externalstress=externalstress,
        ttime=ttime,
        pfactor=pfactor,
    )
    # Run NPT dynamics
    md_step_times = []
    pbar = tqdm(range(steps), desc=f"NPT at {temperature} K")
    for i in range(steps):
        time1 = time.time()
        if i == 0:
            dyn.run(1)
        else:
            dyn.step()
        md_step_times.append(time.time() - time1)
        pbar.update(1)
        temp = dyn.atoms.get_temperature()
        pbar.set_postfix({"Temp": temp, "EnvTemp": temperature})
        if i % log_interval == 0:
            partition_time = calc.partition_times[-1]
            forward_time = calc.forward_times[-1]
            collect_time = calc.collect_times[-1]
            partition_size = calc.partition_sizes[-1]
            wandb.log(
                {
                    "Temperature (K)": temp,
                    "Volume (Å³)": atoms.get_volume(),
                    "Time (fs)": (i+step_bias) * timestep,
                    "Partition Time (s)": partition_time,
                    "Forward Time (s)": forward_time,
                    "Collect Time (s)": collect_time,
                    "Partition Size (atoms)": partition_size,
                    "MD Step Time (s)": md_step_times[-1],
                    "MD Overall Time (s)": time.time() - time1,
                },
                step=i+step_bias,
            )
        if i % save_interval == 0:
            pos = np.array(dyn.atoms.get_positions())
            atomic_numbers = np.array(dyn.atoms.get_atomic_numbers())
            cell = np.array(dyn.atoms.get_cell(complete=True))
            np.savez(
                os.path.join(save_dir, f"md_step_{i+step_bias}.npz"),
                pos=pos,
                atomic_numbers=atomic_numbers,
                cell=cell,
            )
        if i % 100 == 99:
            scaled_pos = np.mod(atoms.get_scaled_positions(), 1.0)
            dyn.atoms.set_scaled_positions(scaled_pos)
    pbar.close()
    return atoms


def main(args_dict: dict):
    model_name = args_dict["ckpt_path"].split("/")[-1].replace(".ckpt", "")
    args_dict["mp_steps"] = int(model_name.split("-MPx")[-1][0])
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
    
    atoms:Atoms = read(args_dict["init_struct"]) # type: ignore
    system = f"SiO-{len(atoms)}"
    atoms.pbc = True
    atoms.calc = calc
    
    wandb.init(
        project="MatterTune-MD-with-Partition",
        name=f"{system}_{model_name}",
        save_code=False,
        config=args_dict,
    )
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d %H:%M")
    save_dir = os.path.join(args_dict["save_dir"], f"{system}_{formatted_time}_{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    total_num_steps = 0
    
    ## 10ps @ 6000K Langevin
    atoms = run_langevin(
        atoms=atoms,
        calc=calc,
        temperature=4000.0,
        timestep=args_dict["timestep"],
        steps=int(10 * 1000 / args_dict["timestep"]),  # 10 ps
        friction=0.01/units.fs,
        log_interval=args_dict["log_interval"],
        save_interval=args_dict["save_interval"],
        save_dir=save_dir,
    )
    total_num_steps += int(10 * 1000 / args_dict["timestep"])
    
    ## 100ps @ 2300K NPT
    PFACTOR = (args_dict["p_time"] * 1000 * units.fs)**2 * (args_dict["bulk_modulus"] * units.eV / (1.0 * units.Ang**3))  # in eV/Ang^3
    TTIME = args_dict["ttime"] * units.fs  # in fs
    atoms = run_npt(
        atoms=atoms,
        calc=calc,
        temperature=2300.0,
        timestep=args_dict["timestep"],
        steps=int(100 * 1000 / args_dict["timestep"]),  # 100 ps
        externalstress=0,
        ttime=TTIME,
        pfactor=PFACTOR,
        log_interval=args_dict["log_interval"],
        save_interval=args_dict["save_interval"],
        save_dir=save_dir,
        step_bias=total_num_steps,
    )
    total_num_steps += int(100 * 1000 / args_dict["timestep"])
    
    ## quenching 2300k->300K NPT
    quench_speed = args_dict["quench_speed"] * 1e-15 # from K/s to K/fs
    quench_steps = int((2300 - 300) / quench_speed / args_dict["timestep"])  # steps to quench from 2300K to 300K
    quench_block_size = args_dict["quench_block_size"]  # number of steps to quench at once
    num_blocks = quench_steps // quench_block_size  # number of blocks to quench
    for block_id in range(num_blocks):
        T_now = 2300.0 - (block_id+1) * quench_block_size * args_dict["timestep"] * quench_speed
        run_npt(
            atoms=atoms,
            calc=calc,
            temperature=T_now,
            timestep=args_dict["timestep"],
            steps=quench_block_size,
            externalstress=0,
            ttime=TTIME,
            pfactor=PFACTOR,
            log_interval=args_dict["log_interval"],
            save_interval=args_dict["save_interval"],
            save_dir=save_dir,
            step_bias=total_num_steps,
        )
        total_num_steps += quench_block_size

    ## 10ps @ 300K NPT
    atoms = run_npt(
        atoms=atoms,
        calc=calc,
        temperature=300.0,
        timestep=args_dict["timestep"],
        steps=int(10 * 1000 / args_dict["timestep"]),  # 10 ps
        externalstress=0,
        ttime=TTIME,
        pfactor=PFACTOR,
        log_interval=args_dict["log_interval"],
        save_interval=args_dict["save_interval"],
        save_dir=save_dir,
        step_bias=total_num_steps,
    )
    total_num_steps += int(10 * 1000 / args_dict["timestep"])
    
    inferencer.exit()
    
if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="../early_stop_mp/SiOx-checkpoints-5A-4A/mattersim-1m-best-2%-MPx2.ckpt",
    )
    parser.add_argument("--init_struct", type=str, default="./init_structs/Si-O_43904.xyz")
    parser.add_argument("--granularity", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--timestep", type=float, default=4) # in fs
    parser.add_argument("--ttime", type=float, default=100) # in fs
    parser.add_argument("--p_time", type=float, default=5) # in ps
    parser.add_argument("--bulk_modulus", type=float, default=0.25)  # in eV/Ang^3
    parser.add_argument("--quench_speed", type=float, default=5e12)  # K/s
    parser.add_argument("--quench_block_size", type=int, default=1000)  # quench 1000 steps at once
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="/storage/lingyu/md_with_partition/")
    parser.add_argument("--tmp_dir", type=str, default="/storage/lingyu/tmp/")
    args_dict = vars(parser.parse_args())
    
    set_global_random_seed(42)  # Set a global random seed for reproducibility    

    main(args_dict)