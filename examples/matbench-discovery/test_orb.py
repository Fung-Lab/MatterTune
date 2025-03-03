from __future__ import annotations

import logging
from pathlib import Path
import time
import rich
import os
import random

import ase
import ase.io as ase_io
import pandas as pd
import numpy as np
import torch
from ase.calculators.calculator import Calculator
from ase.filters import Filter, FrechetCellFilter, UnitCellFilter, ExpCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from matbench_discovery.data import DataFiles, ase_atoms_from_zip
from matbench_discovery.energy import get_e_form_per_atom
from matbench_discovery.enums import MbdKey
from matbench_discovery.metrics.discovery import stable_metrics
from matbench_discovery.plots import wandb_scatter
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key
from tqdm import tqdm
import multiprocessing
import wandb

from orb_models.forcefield.calculator import ORBCalculator
from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
import mattertune.configs as MC
from mattertune.wrappers.ase_calculator import MatterTuneCalculator

torch.set_float32_matmul_precision("high")
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

FILTER_CLS: dict = {"frechet": FrechetCellFilter, "unit": UnitCellFilter, "exp": ExpCellFilter}
OPTIM_CLS: dict = {"FIRE": FIRE, "LBFGS": LBFGS, "BFGS": BFGS}

SETTINGS = {
    "eqv2" : {
        "optimizer": "FIRE",
        "filter": "exp",
        "fmax": 0.02,
        "steps": 500,
    },
    "orb" : {
        "optimizer": "FIRE",
        "filter": "exp",
        "fmax": 0.05,
        "steps": 500,
    },
    "mattersim" : {
        "optimizer": "FIRE",
        "filter": "exp",
        "fmax": 0.02,
        "steps": 500,
    },
}


def load_model_to_calculator(
    model_type: str,
    device: int,
):
    model = ORB_PRETRAINED_MODELS["orb-v2"]()
    device = torch.device(f"cuda:{device}")
    model.to(device)
    orb_calc = ORBCalculator(model, device=device)
    return orb_calc

def relax_atoms_list(
    atoms_list: list[ase.Atoms],
    calculator: Calculator,
    optimizer: str,
    filter: str | None = None,
    fmax: float = 0.02,
    max_steps: int = 500,
):
    relaxed_atoms_list = []
    for atoms in atoms_list:
        time1 = time.time()
        assert type(atoms) == ase.Atoms
        atoms.calc = calculator
        if filter is not None:
            ecf = FILTER_CLS[filter](atoms) # type: ignore
        else:
            ecf = atoms
        optimizer_cls = OPTIM_CLS[optimizer]
        opt = optimizer_cls(ecf, logfile="-")  # type: ignore
        opt.run(fmax=fmax, steps=max_steps)
        if opt.get_number_of_steps() == max_steps:
            atoms.info["converged"] = False
        else:
            atoms.info["converged"] = True
        relaxed_atoms_list.append(atoms)
        time2 = time.time()
        steps = opt.get_number_of_steps()
        wandb.log({
            "Relax time": time2 - time1,
            "Number of steps": steps,
        })
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
    mat_id_to_ehull = dict(zip(df_wbm["material_id"], df_wbm[MbdKey.each_true]))
    mat_id_to_e = dict(zip(df_wbm["material_id"], df_wbm[MbdKey.dft_energy]))

    def parse_single_atoms(atoms: ase.Atoms):
        structure = AseAtomsAdaptor.get_structure(atoms)  # type: ignore
        energy = atoms.info["energy"]
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
        e_hull_true = mat_id_to_ehull[mat_id]
        e_hull_pred = e_hull_true + e_form_pred - e_form_true
        
        e_true = mat_id_to_e[mat_id]

        return mat_id, converged, e_form_pred, e_form_true, e_hull_pred, e_hull_true, energy, corrected_energy, e_true


    for atoms in tqdm(atoms_list, "Processing relaxed structures"):
        mat_id, converged, e_form_pred, e_form_true, e_hull_pred, e_hull_true, energy, corrected_energy, e_true= (
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
        atoms.info["true_energy"] = e_true

    return atoms_list


def single_process_relax(args_dict: dict, atoms_list: list[ase.Atoms]):
    calculator = load_model_to_calculator(
        model_type=args_dict["model_type"],
        device=args_dict["device"],
    )
    relaxed_atoms_list = relax_atoms_list(
        atoms_list=atoms_list,
        calculator=calculator,
        optimizer=SETTINGS[args_dict["model_type"]]["optimizer"], # type: ignore
        filter=SETTINGS[args_dict["model_type"]]["filter"], # type: ignore
        fmax=SETTINGS[args_dict["model_type"]]["fmax"], # type: ignore
        max_steps=SETTINGS[args_dict["model_type"]]["steps"], # type: ignore
    )
    
    # Remove calculator and store infomation in atoms.info
    for atoms in relaxed_atoms_list:
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        atoms.calc = None
    return relaxed_atoms_list

def main(args_dict: dict):
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
    
    # Relax structures with multiprocessing
    start_time = time.time()
    n_jobs = args_dict["n_jobs"]
    n_jobs = max(min(n_jobs, multiprocessing.cpu_count()), 1)
    rich.print(f"Using {n_jobs} jobs")
    if n_jobs == 1:
        relaxed_atoms_list = single_process_relax(args_dict, init_wbm_atoms_list)
    else:
        chunk_size = len(init_wbm_atoms_list) // n_jobs
        chunks = [init_wbm_atoms_list[i:i + chunk_size] for i in range(0, len(init_wbm_atoms_list), chunk_size)]
        with multiprocessing.Pool(n_jobs) as pool:
            results = pool.starmap(single_process_relax, [(args_dict, chunk) for chunk in chunks])
        relaxed_atoms_list = [atoms for chunk in results for atoms in chunk]
    end_time = time.time()
    rich.print(f"Relaxation of {len(init_wbm_atoms_list)} structures took {end_time - start_time:.2f} seconds")
    
    # Save results
    relax_atoms_list_with_info = parse_relaxed_atoms_list(relaxed_atoms_list)
    save_dir = f"{args_dict['save_dir']}/{args_dict['model_type']}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/relaxed_atoms_{l_idx}_{r_idx}.xyz"
    ase_io.write(save_path, relax_atoms_list_with_info)
        
    e_hull_preds = np.array([atoms.info["e_hull_pred"] for atoms in relax_atoms_list_with_info])
    e_hull_trues = np.array([atoms.info["e_hull_true"] for atoms in relax_atoms_list_with_info])
    
    rich.print(stable_metrics(e_hull_trues, e_hull_preds))
    wandb.save(save_path)
    
    e_preds = np.array([atoms.info["corrected_energy"] for atoms in relax_atoms_list_with_info])
    e_trues = np.array([atoms.info["true_energy"] for atoms in relax_atoms_list_with_info])
    
    e_form_preds = np.array([atoms.info["e_form_pred"] for atoms in relax_atoms_list_with_info])
    e_form_trues = np.array([atoms.info["e_form_true"] for atoms in relax_atoms_list_with_info])
    
    data_frame = pd.DataFrame(
        {
            "e_form_pred": e_form_preds,
            "e_form_true": e_form_trues,
            "e_hull_pred": e_hull_preds,
            "e_hull_true": e_hull_trues,
            "e_pred": e_preds,
            "e_true": e_trues,
        }
    )
    table = wandb.Table(dataframe=data_frame)
    wandb_scatter(
        table=table,
        fields=dict(x="e_form_true", y="e_form_pred"),
        title=f"{args_dict['model_type']} e_form {l_idx}-{r_idx}",
    )
    wandb_scatter(
        table=table,
        fields=dict(x="e_hull_true", y="e_hull_pred"),
        title=f"{args_dict['model_type']} e_hull {l_idx}-{r_idx}",
    )
    wandb_scatter(
        table=table,
        fields=dict(x="e_true", y="e_pred"),
        title=f"{args_dict['model_type']} e {l_idx}-{r_idx}",
    )
    
    if args_dict["delete_files"]:
        os.system(f"rm -rf {save_path}")
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="eqv2")
    parser.add_argument("--l_idx", type=int, default=0)
    parser.add_argument("--r_idx", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="/net/csefiles/coc-fung-cluster/lingyu/matbench-discovery")
    parser.add_argument("--show_log", action='store_true')
    parser.add_argument("--delete_files", action='store_true')
    args = parser.parse_args()
    args_dict = vars(args)

    main(args_dict)