from __future__ import annotations

import copy
import time
import rich
import logging
from tqdm import tqdm

import ase.units as units
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.npt import NPT
import wandb

from mattertune.students import (
    CACEStudentModel,
    SchNetStudentModel,
)
from mattertune.backbones import (
    MatterSimM3GNetBackboneModule,
    MACEBackboneModule,
    ORBBackboneModule,
    UMABackboneModule,
)
from mattertune.util import set_global_random_seed
from mattertune.wrappers.ase_calculator import MatterTunePartitionCalculator
from mattertune.wrappers.utils.parallel_inference import ParallizedInferenceDDP

logging.basicConfig(level=logging.ERROR)

# supercell_as = [8, 10, 14, 18]
supercell_as = [18]

def main(args_dict: dict):
    
    md_atoms_base = read("./data/H2O.xyz")
    assert isinstance(md_atoms_base, Atoms), "Expected an Atoms object"
    md_atoms_base.pbc = True
    
    model_path = args_dict["model"]
    model_type = model_path.split("/")[-1].replace(".ckpt", "").lower()
    if "cace" in model_type:
        model = CACEStudentModel.load_from_checkpoint(model_path, lazy_init_atoms=md_atoms_base)
    elif "schnet" in model_type:
        model = SchNetStudentModel.load_from_checkpoint(model_path)
    elif "mattersim" in model_type:
        model = MatterSimM3GNetBackboneModule.load_from_checkpoint(model_path, map_location="cpu")
    elif "mace" in model_type:
        model = MACEBackboneModule.load_from_checkpoint(model_path, map_location="cpu")
    elif "orb" in model_type:
        model = ORBBackboneModule.load_from_checkpoint(model_path, map_location="cpu")
    elif "uma" in model_type:
        model = UMABackboneModule.load_from_checkpoint(model_path, map_location="cpu")
    else:
        raise NotImplementedError()
    inferencer = ParallizedInferenceDDP(
        ckpt_path=args_dict["model"],
        properties=["energy", "forces"],
        devices=args_dict["devices"],
        batch_size=1,
        num_workers=args_dict["num_workers"],
        using_partition=True,
        tmp_dir="./tmp",
        lazy_init_atoms="./data/H2O.xyz" if "cace" in model_type else None,
    )
    calc = MatterTunePartitionCalculator(
        model=model,
        inferencer=inferencer,
        mp_steps=args_dict["mp_steps"],
        granularity=(2, 2, 2)
    )

    wandb.init(
        project="MatterTune-Distill-MDSpeed",
        name=f"{model_type}-H2O-298K-{args_dict['thermo_state']}-devices{'-'.join(map(str, args_dict['devices']))}-mp{args_dict['mp_steps']}",
        save_code=False,
    )
    wandb.config.update(args_dict)
    
    idx = 0
    while idx < len(supercell_as):
        a = supercell_as[idx]
        md_atoms = copy.deepcopy(md_atoms_base)
        md_atoms = md_atoms * (a, a, a)
        md_atoms.calc = calc
        rich.print(f"Running supercell {a}x{a}x{a}, {len(md_atoms)} atoms, Partition Granularity: {calc.granularity}")
    
        if args_dict["thermo_state"].lower() == "nvt":
            dyn = Langevin(
                md_atoms,
                temperature_K=298,
                timestep=args_dict["timestep"] * units.fs,
                friction=args_dict["friction"],
            )
        elif args_dict["thermo_state"].lower() == "npt":
            dyn = NPT(
                md_atoms,
                temperature_K=298,  # 300 K
                timestep=args_dict["timestep"] * units.fs,  # 0.5 fs
                externalstress=None,
                ttime=100 * units.fs,
                pfactor=None,
            )
        else:
            raise ValueError("Invalid thermo_state, must be one of 'NVT' or 'NPT'")
        
        time1 = time.time()
        try:
            for step_i in tqdm(range(args_dict["steps"])):
                if step_i == 0:
                    dyn.run(1)
                else:
                    dyn.step()
                wandb.log({
                    "Temperature (K)": dyn.atoms.get_temperature(),
                    "Partition Size": calc.last_partition_size,
                    "Extra Time (s)": calc.last_extra_time,
                    "Partition Times (s)": calc.last_partition_time,
                    "Forward Times (s)": calc.last_forward_time,
                    "Collect Times (s)": calc.last_collect_time,
                    "Denormalization Times (s)": calc.last_denormalize_time,
                })
        except Exception as e:
            # check if it is CUDA out of memory error
            if "CUDA out of memory" in str(e):
                rich.print(f"[red]CUDA out of memory for supercell {a}x{a}x{a} under granularity: {calc.granularity}")
                calc.increase_granularity(1)
                rich.print(f"[red]Increase granularity to {calc.granularity} and retrying...")
                continue
            else:
                raise e
        time2 = time.time()
        
        ps_per_hour = (args_dict["steps"] * args_dict["timestep"]) / (time2 - time1) * 3600 / 1000
        rich.print(f"[green]Supercell {a}x{a}x{a}, {len(md_atoms)} atoms, {ps_per_hour:.2f} ps/hour")
        wandb.log({f"Speed (ps/hour) {a}x{a}x{a}": ps_per_hour})
        rich.print("=" * 50)
        idx += 1
    
    wandb.finish()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./checkpoints/schnet-5.0A-T=3.ckpt")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mp_steps", type=int, default=3)
    parser.add_argument("--thermo_state", type=str, default="NVT")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--timestep", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=1000)
    args_dict = vars(parser.parse_args())
    main(args_dict)
