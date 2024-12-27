from __future__ import annotations

import json
import logging

import nshutils as nu
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

import mattertune.configs as MC
from mattertune.normalization import compute_per_atom_references

nu.pretty()

xyz_path = "./data/water_1000_eVAng.xyz"

# atoms_list = read(xyz_path, index=":")
# for i, atoms in enumerate(atoms_list):
#     energy = atoms.info["TotEnergy"]
#     force = np.array(atoms.arrays["force"])
#     calc = SinglePointCalculator(atoms, energy=energy, forces=force)
#     atoms.set_calculator(calc)
#     _energy = atoms.get_potential_energy()
#     _force = atoms.get_forces()
#     assert np.allclose(_energy, energy)
#     assert np.allclose(_force, force)
#     atoms_list[i] = atoms
# write("./data/water_1000_eVAng.xyz", atoms_list)

dataset_config = MC.XYZDatasetConfig(src=xyz_path)
dataset = dataset_config.create_dataset()

ref_dict = compute_per_atom_references(
    dataset=dataset,
    property=MC.EnergyPropertyConfig(loss=MC.MAELossConfig()),
    reference_model="ridge",
)

filename = xyz_path.split("/")[-1].split(".")[0]
json.dump(ref_dict, open(f"./data/{filename}-energy_reference.json", "w"), indent=4)
logging.info(f"Saved energy reference to energy_reference.json")
