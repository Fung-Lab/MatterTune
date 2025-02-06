from __future__ import annotations

import logging
from pathlib import Path
import time
import rich

import ase
import pandas as pd
import numpy as np
from ase.constraints import ExpCellFilter
from ase.optimize import FIRE
from matbench_discovery import today
from matbench_discovery.data import DataFiles, ase_atoms_from_zip
from matbench_discovery.energy import get_e_form_per_atom
from matbench_discovery.enums import MbdKey
from matbench_discovery.metrics.discovery import stable_metrics
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key
from tqdm import tqdm

import mattertune.configs as MC
from mattertune.wrappers.ase_calculator import MatterTuneCalculator

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


def load_model_to_calculator(
    model_type: str,
    device: int,
):
    model_type = model_type.lower()

    if "eqv2" in model_type:
        model_config = MC.EqV2BackboneConfig.draft()
        model_config.checkpoint_path = Path(
            "/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/eqV2_31M_mp.pt"
        )
        model_config.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig.draft()
        model_config.atoms_to_graph.radius = 8.0
        model_config.atoms_to_graph.max_num_neighbors = 20
    elif "jmp" in model_type:
        model_config = MC.JMPBackboneConfig.draft()
        model_config.graph_computer = MC.JMPGraphComputerConfig.draft()
        model_config.graph_computer.pbc = True
        model_config.pretrained_model = model_type
    elif "orb" in model_type:
        model_config = MC.ORBBackboneConfig.draft()
        model_config.pretrained_model = model_type
    elif "mattersim" in model_type:
        model_config = MC.MatterSimBackboneConfig.draft()
        model_config.graph_convertor = graph_convertor = (
            MC.MatterSimGraphConvertorConfig.draft()
        )
        if model_type == "mattersim-1m":
            model_config.pretrained_model = "MatterSim-v1.0.0-1M"
        elif model_type == "mattersim-5m":
            model_config.pretrained_model = "MatterSim-v1.0.0-5M"
        else:
            raise ValueError(f"Model type {model_type} not recognized")
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
            "precision": "32",
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
    fmax: float = 0.02,
    steps: int = 500,
):
    relaxed_atoms_list = []
    pbar = tqdm(atoms_list, desc="Relaxing structures")
    for atoms in atoms_list:
        assert type(atoms) == ase.Atoms
        atoms.set_calculator(calculator)
        ecf = ExpCellFilter(atoms)
        opt = FIRE(ecf)  # type: ignore
        opt.run(fmax=fmax, steps=steps)
        if opt.get_number_of_steps() == steps:
            atoms.info["converged"] = False
        else:
            atoms.info["converged"] = True
        relaxed_atoms_list.append(atoms)
        pbar.update(1)
    pbar.close()
    return relaxed_atoms_list


def parse_relaxed_atoms_list_as_df(
    atoms_list: list[ase.Atoms], *, keep_unconverged: bool = True
) -> pd.DataFrame:
    e_form_col = "e_form_per_atom"

    wbm_cse_paths = DataFiles.wbm_computed_structure_entries.path
    df_cse = pd.read_json(wbm_cse_paths).set_index(Key.mat_id)

    df_cse[Key.computed_structure_entry] = [
        ComputedStructureEntry.from_dict(dct)
        for dct in tqdm(df_cse[Key.computed_structure_entry], desc="Hydrate CSEs")
    ]

    print(f"Found {len(df_cse):,} CSEs in {wbm_cse_paths=}")
    print(f"Found {len(atoms_list):,} relaxed structures")

    def parse_single_atoms(atoms: ase.Atoms) -> tuple[str, bool, float, float, float]:
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
        formation_energy = (
            get_e_form_per_atom(processed)
            if processed is not None
            else get_e_form_per_atom(cse)
        )

        return mat_id, converged, formation_energy, energy, corrected_energy

    mat_id_list, converged_list, e_form_list = [], [], []
    energy_list, corrected_energy_list = [], []

    for atoms in tqdm(atoms_list, "Processing relaxed structures"):
        mat_id, converged, formation_energy, energy, corrected_energy = (
            parse_single_atoms(atoms)
        )
        if not keep_unconverged and not converged:
            continue
        mat_id_list += [mat_id]
        converged_list += [converged]
        e_form_list += [formation_energy]
        energy_list += [energy]
        corrected_energy_list += [corrected_energy]

    return pd.DataFrame(
        {
            Key.mat_id: mat_id_list,
            "converged": converged_list,
            e_form_col: e_form_list,
            "model_energy": energy_list,
            "corrected_energy": corrected_energy_list,
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mattersim-1m")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    
    # Load the model and get the calculator
    calculator = load_model_to_calculator(
        model_type=args_dict["model_type"],
        device=args_dict["device"],
    )
    # Load the initial structures
    init_wbm_atoms_list: list[ase.Atoms] = ase_atoms_from_zip(
        DataFiles.wbm_initial_atoms.path
    )
    init_wbm_atoms_list = init_wbm_atoms_list[:2]
    # Relax
    start_time = time.time()
    relaxed_atoms_list = relax_atoms_list(
        atoms_list=init_wbm_atoms_list,
        calculator=calculator,
    )
    end_time = time.time() - start_time
    print(f"Relaxation took {end_time:.2f} seconds")

    # Results analysis
    result_df = parse_relaxed_atoms_list_as_df(relaxed_atoms_list)
    e_form_pred = np.array(result_df["e_form_per_atom"])
    df_wbm = pd.read_csv(DataFiles.wbm_summary.path)
    e_form_true = np.array(df_wbm[MbdKey.e_form_dft.label])
    e_hull_true = np.array(df_wbm[MbdKey.each_true.label])
    e_hull_pred = e_hull_true + e_form_pred - e_form_true
    e_hull_pred = e_hull_pred.tolist()
    e_hull_true = e_hull_true.tolist()
    
    rich.print(stable_metrics(e_hull_true, e_hull_pred))
    
    # result_df.to_csv(
    #     f"{today}-{args_dict['model_type']}-wbm-IS2RE.csv.gz"
    # )
    
