from __future__ import annotations

import rich
import time
import copy

from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
import ase.units as units
from ase.md.npt import NPT
import wandb

from mattertune.backbones import (
    JMPBackboneModule,
    MatterSimM3GNetBackboneModule,
    ORBBackboneModule,
    EqV2BackboneModule,
    MACEBackboneModule,
    UMABackboneModule,
    NequIPBackboneModule,
)



def main(args_dict: dict):
    
    atoms_prototype:Atoms = read("./data/H2O.xyz") # type: ignore
    atoms_prototype.pbc = True
    
    
    if not args_dict["compiled"]:
        model_name = args_dict["ckpt_path"].split("/")[-1].replace(".ckpt", "").lower()
        if "jmp" in model_name:
            model = JMPBackboneModule.load_from_checkpoint(
                checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
            )
        elif "orb" in model_name:
            model = ORBBackboneModule.load_from_checkpoint(
                checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
            )
        elif "mattersim" in model_name:
            model = MatterSimM3GNetBackboneModule.load_from_checkpoint(
                checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
            )
        elif "eqv2" in model_name:
            model = EqV2BackboneModule.load_from_checkpoint(
                checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
            )
        elif "mace" in model_name:
            model = MACEBackboneModule.load_from_checkpoint(
                checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
            )
        elif "uma" in model_name:
            model = UMABackboneModule.load_from_checkpoint(
                checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
            )
        elif "nequip" in model_name:
            model = NequIPBackboneModule.load_from_checkpoint(
                checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
            )
        else:
            raise ValueError(
                "Invalid fine-tuning model name"
            )
    
        calculator = model.ase_calculator(device=f"cuda")
    else:
        from nequip.ase import NequIPCalculator
        
        calculator = NequIPCalculator.from_compiled_model(
            compile_path=args_dict["ckpt_path"],
            device=f"cuda",
            chemical_species_to_atom_type_map=True, 
        )
        model_name = args_dict["ckpt_path"].split("/")[-1].split(".")[0].lower()
    
    wandb.init(
        project="MatterTune-Speed-Test-Main",
        name=f"{model_name}-{args_dict['dyn']}-cuda{args_dict['device']}",
        config=args_dict,
    )
    
    rich.print(f"Model: {model_name}")
    rich.print("Device:", args_dict["device"])
    rich.print("Dynamics Type:", args_dict["dyn"])
    
    for a in range(1, 8):
        atoms = copy.deepcopy(atoms_prototype)
        atoms *= (a, a, a)
        try:
            atoms.calc = calculator
            
            if args_dict["dyn"] == "langevin":
                dyn = Langevin(
                    atoms,
                    timestep=1.0 * units.fs,
                    temperature_K=300,
                    friction=0.02,
                )
            elif args_dict["dyn"] == "npt":
                dyn = NPT(
                    atoms,
                    timestep=1.0 * units.fs,
                    temperature_K=300,
                    externalstress=None,
                    ttime=100 * units.fs,
                    pfactor=None,
                )
            else:
                raise ValueError("Invalid dyn type, must be one of 'langevin' or 'npt'")
            
            time1 = time.time()
            dyn.run(100)
            time2 = time.time()
            
            rich.print(f"Number of atoms: {len(atoms)}")
            rich.print(f"Time taken for 100 steps: {time2 - time1} seconds")
            rich.print(f"Average time per step: {(time2 - time1) / 100} seconds")
            us_per_atom_step = ((time2 - time1) / 100) * 1e6 / len(atoms)
            rich.print(f"Microseconds per atom per step: {us_per_atom_step} µs/atom·step")
            rich.print("=" * 40)
            
            wandb.log({
                "num_atoms": len(atoms),
                "time_per_100_steps": time2 - time1,
                "time_per_step": (time2 - time1) / 100,
                "us_per_atom_step": us_per_atom_step,
                f"Time Per Step {len(atoms)}-atoms": (time2 - time1) / 100,
            })
            
        except RuntimeError as e:
            # check for out-of-memory error
            if "out of memory" in str(e):
                rich.print(f"Out of memory for {len(atoms)} atoms, stopping test.")
                break
            else:
                raise e
            
    wandb.finish()
        
        
if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dyn",
        type=str,
        default="langevin",
        choices=["langevin", "npt"],
    )
    parser.add_argument(
        "--compiled",
        action="store_true",
    )
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)