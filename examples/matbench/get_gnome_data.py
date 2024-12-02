from __future__ import annotations

import copy
import os
import shutil
import tempfile
import zipfile

import pandas as pd
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm

## Download the data from the public link
PUBLIC_LINK = "https://storage.googleapis.com/"
BUCKET_NAME = "gdm_materials_discovery"
FOLDER_NAME = "gnome_data"
FILES = (
    "stable_materials_summary.csv",
    "by_reduced_formula.zip",
)
PROPERTY: str | None = "Bandgap"


def download_from_link(link: str, output_dir: str):
    """Download a file from a public link using wget."""
    os.system(f"wget {link} -P {output_dir}")


parent_directory = os.path.join(PUBLIC_LINK, BUCKET_NAME)
for filename in FILES:
    if not os.path.exists(filename):
        public_link = os.path.join(parent_directory, FOLDER_NAME, filename)
        download_from_link(public_link, ".")

# Read the contents of the zipfile
# Due to the size of the release, this takes about 30 seconds but prevents
# the need to extract all files in order to read individual structures
z = zipfile.ZipFile("by_reduced_formula.zip")
gnome_crystals = pd.read_csv("stable_materials_summary.csv", index_col=0)

## Extract all colomn indices with Bandgap not NaN and inf
if PROPERTY is not None:
    gnome_crystals = gnome_crystals[gnome_crystals[PROPERTY].notna()]
else:
    all_properties = gnome_crystals.columns
    print("All properties available in the dataset:", all_properties)
    exit()

## Extract their Reduced Formula
reduced_formulas = gnome_crystals["Reduced Formula"]


## Extract the corresponding structures
def obtain_structure(
    reduced_formula: str,
) -> Atoms:
    """Obtain the structure from a provided reduced formula."""
    temp_dir = tempfile.TemporaryDirectory()
    extension = f"{reduced_formula}.CIF"
    temp_path = os.path.join(temp_dir.name, extension)

    with z.open(os.path.join("by_reduced_formula", extension)) as zf:
        with open(temp_path, "wb") as fp:
            shutil.copyfileobj(zf, fp)

    atoms = read(temp_path)
    assert isinstance(atoms, Atoms), "Expected an Atoms object"
    temp_dir.cleanup()
    return atoms


all_atoms = []
pbar = tqdm(total=len(reduced_formulas))
for formula in reduced_formulas:
    atoms_i = obtain_structure(formula)
    property = gnome_crystals.loc[gnome_crystals["Reduced Formula"] == formula][
        PROPERTY
    ].values[0]
    try:
        property = float(property)
    except ValueError:
        print(f"Property {PROPERTY} is not a float for {formula}, found {property}")
        continue
    atoms_i.info[PROPERTY.lower()] = property
    # symbols = atoms_i.get_chemical_symbols()
    # pos = atoms_i.get_positions()
    # cell = atoms_i.get_cell(complete=True)
    all_atoms.append(copy.deepcopy(atoms_i))
    pbar.update(1)
    pbar.set_description(f"{PROPERTY}: {property:.4f}")

os.makedirs("./data", exist_ok=True)
write(f"./data/gnome_{PROPERTY}.xyz", all_atoms)
