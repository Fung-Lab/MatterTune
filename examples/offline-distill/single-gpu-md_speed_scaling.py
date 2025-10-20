from __future__ import annotations

import copy
import time
import rich

import ase.units as units
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.npt import NPT
import wandb
from tqdm import tqdm

from mattertune.students import (
    CACEStudentModel,
    SchNetStudentModel,
    PaiNNStudentModel,
)
from mattertune.backbones import (
    MatterSimM3GNetBackboneModule,
    MACEBackboneModule,
    ORBBackboneModule,
    UMABackboneModule,
)

supercell_as = [1, 2, 3, 4, 5, 8, 10]


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
    elif "painn" in model_type:
        model = PaiNNStudentModel.load_from_checkpoint(model_path)
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
    calc = model.ase_calculator(device=f"cuda:{args_dict['device']}")

    wandb.init(
        project="MatterTune-Distill-MDSpeed",
        name=f"{model_type}-H2O-298K-{args_dict['thermo_state']}",
        save_code=False,
    )
    wandb.config.update(args_dict)
    
    for a in supercell_as:
        md_atoms = copy.deepcopy(md_atoms_base)
        md_atoms = md_atoms * (a, a, a)
        md_atoms.calc = calc
        rich.print(f"Running supercell {a}x{a}x{a}, {len(md_atoms)} atoms")
    
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
                    "Build Graph Time (s)": calc.last_build_graph_time,
                    "Forward Time (s)": calc.last_forward_time,
                    "Calculate Time (s)": calc.last_calculation_time,
                })
        except Exception as e:
            # check if it is CUDA out of memory error
            if "CUDA out of memory" in str(e):
                rich.print(f"[red]CUDA out of memory for supercell {a}x{a}x{a}, skipping...[/red]")
                continue
            else:
                raise e 
        time2 = time.time()
        
        ps_per_hour = (args_dict["steps"] * args_dict["timestep"]) / (time2 - time1) * 3600 / 1000
        rich.print(f"Supercell {a}x{a}x{a}, {len(md_atoms)} atoms, {ps_per_hour:.2f} ps/hour")
        wandb.log({f"Speed (ps/hour) {a}x{a}x{a}": ps_per_hour})
        rich.print("=" * 50)
    
    wandb.finish()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./checkpoints/cace-5.5A-T=1.ckpt")
    parser.add_argument("--thermo_state", type=str, default="NVT")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--timestep", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=1000)
    args_dict = vars(parser.parse_args())
    main(args_dict)
