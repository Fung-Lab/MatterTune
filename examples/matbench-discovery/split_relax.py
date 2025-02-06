from __future__ import annotations

import logging
from pathlib import Path
import time
import rich
import os

import ase
import ase.io as ase_io
import pandas as pd
import numpy as np
from ase.filters import Filter, FrechetCellFilter, UnitCellFilter, ExpCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from matbench_discovery.data import DataFiles, ase_atoms_from_zip
from matbench_discovery.energy import get_e_form_per_atom
from matbench_discovery.enums import MbdKey
from matbench_discovery.metrics.discovery import stable_metrics
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key
from tqdm import tqdm
import wandb

import mattertune.configs as MC
from mattertune.wrappers.ase_calculator import MatterTuneCalculator


logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

FILTER_CLS: dict = {"frechet": FrechetCellFilter, "unit": UnitCellFilter, "exp": ExpCellFilter}
OPTIM_CLS: dict = {"FIRE": FIRE, "LBFGS": LBFGS, "BFGS": BFGS}


def load_model_to_calculator(
    model_type: str,
    device: int,
):
    model_type = model_type.lower()

    if "eqv2" in model_type:
        model_config = MC.EqV2BackboneConfig.draft()
        model_config.checkpoint_path = Path(
            "./checkpoints/eqV2_dens_31M_mp.pt"
        )
        model_config.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig.draft()
        model_config.atoms_to_graph.radius = 8.0
        model_config.atoms_to_graph.max_num_neighbors = 20
    elif "orb" in model_type:
        model_config = MC.ORBBackboneConfig.draft()
        model_config.pretrained_model = "orb-v2"
    elif "mattersim" in model_type:
        model_config = MC.MatterSimBackboneConfig.draft()
        model_config.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        model_config.pretrained_model = "MatterSim-v1.0.0-5M"
    else:
        raise ValueError(f"Model type {model_type} not recognized")
    model_config.ignore_gpu_batch_transform_error = True
    model_config.freeze_backbone = False
    model_config.optimizer = MC.AdamConfig(lr=1e-3)

    model_config.reset_output_heads = False
    conservative = True if "mattersim" in model_type else False
    model_config.properties = []
    energy = MC.EnergyPropertyConfig(loss=MC.MSELossConfig())
    model_config.properties.append(energy)
    forces = MC.ForcesPropertyConfig(loss=MC.MSELossConfig(), conservative=conservative)
    model_config.properties.append(forces)
    stresses = MC.StressesPropertyConfig(
        loss=MC.MSELossConfig(), conservative=conservative
    )
    model_config.properties.append(stresses)

    model = model_config.create_model()
    calculator = model.ase_calculator(
        lightning_trainer_kwargs={
            "accelerator": "gpu",
            "devices": [device],
            "precision": "32" if "mattersim" in model_type else "bf16-mixed",
            "inference_mode": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "barebones": True,
        }
    )
    return calculator

def relax_atoms_list(
    atoms_list: list[ase.Atoms],
    calculator: MatterTuneCalculator,
    optimizer_cls: Optimizer,
    filter_cls: Filter | None = None,
    fmax: float = 0.02,
    steps: int = 500,
):
    relaxed_atoms_list = []
    pbar = tqdm(atoms_list, desc="Relaxing structures")
    for atoms in atoms_list:
        assert type(atoms) == ase.Atoms
        atoms.set_calculator(calculator)
        if filter_cls is not None:
            ecf = filter_cls(atoms) # type: ignore
        else:
            ecf = atoms
        opt = optimizer_cls(ecf, logfile="-" if args_dict["show_log"] else None)  # type: ignore
        opt.run(fmax=fmax, steps=steps)
        if opt.get_number_of_steps() == steps:
            atoms.info["converged"] = False
        else:
            atoms.info["converged"] = True
        relaxed_atoms_list.append(atoms)
        pbar.update(1)
    pbar.close()
    return relaxed_atoms_list

def parse_relaxed_atoms_list(
    atoms_list: list[ase.Atoms],
):

    wbm_cse_paths = DataFiles.wbm_computed_structure_entries.path
    df_cse = pd.read_json(wbm_cse_paths).set_index(Key.mat_id)
    df_cse[Key.computed_structure_entry] = [
        ComputedStructureEntry.from_dict(dct)
        for dct in tqdm(df_cse[Key.computed_structure_entry], desc="Hydrate CSEs")
    ]
    df_wbm = pd.read_csv(DataFiles.wbm_summary.path)
    mat_id_to_eform = dict(zip(df_wbm["material_id"], df_wbm[MbdKey.e_form_dft]))
    mat_id_tp_ehull = dict(zip(df_wbm["material_id"], df_wbm[MbdKey.each_true]))

    def parse_single_atoms(atoms: ase.Atoms):
        structure = AseAtomsAdaptor.get_structure(atoms)  # type: ignore
        energy = atoms.get_potential_energy()
        mat_id = atoms.info["material_id"]
        converged = atoms.info["converged"]

        cse = df_cse.loc[mat_id, Key.computed_structure_entry]
        cse._energy = energy  # type: ignore
        cse._structure = structure  # type: ignore

        processed = MaterialsProject2020Compatibility(check_potcar=False).process_entry(
            cse,  # type: ignore
            clean=True,
        )
        corrected_energy = processed.energy if processed is not None else energy
        e_form_pred = (
            get_e_form_per_atom(processed)
            if processed is not None
            else get_e_form_per_atom(cse)
        )
        
        e_form_true = mat_id_to_eform[mat_id]
        e_hull_true = mat_id_tp_ehull[mat_id]
        e_hull_pred = e_hull_true + e_form_pred - e_form_true

        return mat_id, converged, e_form_pred, e_form_true, e_hull_pred, e_hull_true, energy, corrected_energy


    for atoms in tqdm(atoms_list, "Processing relaxed structures"):
        mat_id, converged, e_form_pred, e_form_true, e_hull_pred, e_hull_true, energy, corrected_energy = (
            parse_single_atoms(atoms)
        )
        
        atoms.info[Key.mat_id] = mat_id
        atoms.info["converged"] = converged
        atoms.info["e_form_pred"] = e_form_pred # ev/atom
        atoms.info["e_form_true"] = e_form_true
        atoms.info["e_hull_pred"] = e_hull_pred
        atoms.info["e_hull_true"] = e_hull_true
        atoms.info["model_energy"] = energy
        atoms.info["corrected_energy"] = corrected_energy

    return atoms_list


SETTINGS = {
    "eqv2" : {
        "optimizer": FIRE,
        "filter": None,
        "fmax": 0.02,
        "steps": 500,
    },
    "orb" : {
        "optimizer": FIRE,
        "filter": FrechetCellFilter,
        "fmax": 0.05,
        "steps": 500,
    },
    "mattersim-1m" : {
        "optimizer": FIRE,
        "filter": ExpCellFilter,
        "fmax": 0.02,
        "steps": 500,
    },
    "mattersim-5m" : {
        "optimizer": FIRE,
        "filter": ExpCellFilter,
        "fmax": 0.02,
        "steps": 500,
    },
}


def main(args_dict: dict):
    # Load calculator and initial structures
    calculator = load_model_to_calculator(
        model_type=args_dict["model_type"],
        device=args_dict["device"],
    )
    init_wbm_atoms_list: list[ase.Atoms] = ase_atoms_from_zip(
        DataFiles.wbm_initial_atoms.path
    )
    total_length = len(init_wbm_atoms_list)
    rich.print(f"Found {total_length:,} initial structures")
    l_idx, r_idx = max(args_dict["l_idx"], 0), min(args_dict["r_idx"], total_length)
    if l_idx >= total_length:
        exit()
    init_wbm_atoms_list = init_wbm_atoms_list[l_idx:r_idx]
    rich.print(f"Processing structures {l_idx:,} to {r_idx:,}")
    
    # Relax structures
    wandb.login()
    wandb.init(
        project="Mattertune-matbench-discovery", 
        name = f"{args_dict['model_type']}_{l_idx}_{r_idx}",
        config=args_dict,
    )
    
    start_time = time.time()
    relaxed_atoms_list = relax_atoms_list(
        atoms_list=init_wbm_atoms_list,
        calculator=calculator,
        optimizer_cls=SETTINGS[args_dict["model_type"]]["optimizer"], # type: ignore
        filter_cls=SETTINGS[args_dict["model_type"]]["filter"], # type: ignore
        fmax=SETTINGS[args_dict["model_type"]]["fmax"], # type: ignore
        steps=SETTINGS[args_dict["model_type"]]["steps"], # type: ignore
    )
    end_time = time.time()
    print(f"Relaxation took {end_time - start_time:.2f} seconds")
    
    # Save results
    relax_atoms_list_with_info = parse_relaxed_atoms_list(relaxed_atoms_list)
    save_dir = f"{args_dict['save_dir']}/{args_dict['model_type']}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/relaxed_atoms_{l_idx}_{r_idx}.xyz"
    ase_io.write(save_path, relax_atoms_list_with_info)
    
    for atoms in relax_atoms_list_with_info:
        print(atoms.info)
        print("===============================")
        
    e_hull_preds = np.array([atoms.info["e_hull_pred"] for atoms in relax_atoms_list_with_info])
    e_hull_trues = np.array([atoms.info["e_hull_true"] for atoms in relax_atoms_list_with_info])
    
    rich.print(stable_metrics(e_hull_trues, e_hull_preds))
    
    wandb.save(save_path)
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="eqv2")
    parser.add_argument("--l_idx", type=int, default=0)
    parser.add_argument("--r_idx", type=int, default=260000)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="/net/csefiles/coc-fung-cluster/lingyu/matbench-discovery")
    parser.add_argument("--show_log", action='store_true')
    args = parser.parse_args()
    args_dict = vars(args)

    main(args_dict)
    
    # pip install "git+https://github.com/FAIR-Chem/fairchem.git@omat24#subdirectory=packages/fairchem-core" --no-deps
    # pip install ase "e3nn>=0.5" hydra-core lmdb numba "numpy>=1.26,<2.0" orjson "pymatgen>=2023.10.3" submitit tensorboard "torch==2.4" wandb torch_geometric h5py netcdf4 opt-einsum spglib