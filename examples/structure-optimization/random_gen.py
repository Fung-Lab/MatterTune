from __future__ import annotations

import os

import numpy as np
from ase import Atoms
from ase.io import read, write
from pyxtal import pyxtal
from tqdm import tqdm


def identical_atoms(atoms1: Atoms, atoms2: Atoms) -> bool:
    if len(atoms1) != len(atoms2):
        return False
    if not np.allclose(atoms1.cell, atoms2.cell):
        return False
    scaled_pos1 = atoms1.get_scaled_positions()
    scaled_pos2 = atoms2.get_scaled_positions()
    symbols1 = atoms1.get_chemical_symbols()
    symbols2 = atoms2.get_chemical_symbols()
    ## Permute atoms1 and atoms2 based on scaled positions
    sorted_indices1 = np.lexsort(scaled_pos1.T)
    sorted_indices2 = np.lexsort(scaled_pos2.T)
    symbols1 = [symbols1[i] for i in sorted_indices1]
    symbols2 = [symbols2[i] for i in sorted_indices2]
    scaled_pos1 = scaled_pos1[sorted_indices1]
    scaled_pos2 = scaled_pos2[sorted_indices2]
    ## Check if the two structures are identical
    for i in range(len(atoms1)):
        if symbols1[i] != symbols2[i]:
            return False
        if not np.allclose(scaled_pos1[i], scaled_pos2[i]):
            return False
    return True


def find_existing(atoms: Atoms, existing_atoms: list) -> bool:
    for existing_atoms_i in existing_atoms:
        if identical_atoms(atoms, existing_atoms_i):
            return True
    return False


def main(num_gen: int):
    os.system("rm -rf ./ZnMn2O4_random/*.res")
    formula = {"Zn": 1, "Mn": 2, "O": 4}
    num_success = 0
    pbar = tqdm(total=num_gen)
    existing_atoms = []
    while num_success < num_gen:
        try:
            ## Generate random
            crystal = pyxtal()
            group_num = np.random.randint(1, 230)
            crystal.from_random(
                dim=3,
                group=group_num,
                species=["Zn", "Mn", "O"],
                numIons=[8, 16, 32],
            )

            ## export the structure to cif file
            atoms = crystal.to_ase()
            if not find_existing(atoms, existing_atoms):
                existing_atoms.append(atoms)
                write("./ZnMn2O4_random/{}.res".format(num_success), atoms)
            num_success += 1
            pbar.update(1)
        except:
            pass


if __name__ == "__main__":
    main(1000)
