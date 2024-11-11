from __future__ import annotations

import ase.neighborlist as nl
import numpy as np
from ase import Atoms
from ase.io import read


def rdf_compute(atoms: Atoms, r_max, n_bins, elements=None):
    scaled_pos = atoms.get_scaled_positions()
    atoms.set_scaled_positions(np.mod(scaled_pos, 1))

    num_atoms = len(atoms)
    volume = atoms.get_volume()
    density = num_atoms / volume

    send_indices, receive_indices, distances = nl.neighbor_list(
        "ijd",
        atoms,
        r_max,
        self_interaction=False,
    )

    if elements is not None and len(elements) == 2:
        species = np.array(atoms.get_chemical_symbols())
        indices = np.where(
            np.logical_and(
                species[send_indices] == elements[0],
                species[receive_indices] == elements[1],
            )
        )[0]
        distances = distances[indices]

        num_atoms = (species == elements[0]).sum()
        density = num_atoms / volume

    hist, bin_edges = np.histogram(distances, range=(0, r_max), bins=n_bins)
    rdf_x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_volume = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    rdf_y = hist / (bin_volume * density * num_atoms)

    rdf = np.vstack((rdf_y, rdf_x)).reshape(1, 2, -1)
    return rdf


def main(args_dict: dict):
    md_traj = read(args_dict["md_traj"], ":")
    md_traj = md_traj[-args_dict["n_frames"] :]
    r_max = args_dict["r_max"]
    n_bins = args_dict["n_bins"]
    elements = args_dict["elements"]

    ## Take the mean of the n_frames' RDFs
    rdf_ys = []
    rdf_xs = []
    for atoms in md_traj:
        rdf = rdf_compute(atoms, r_max, n_bins, elements)
        rdf_ys.append(rdf[0, 0])
        rdf_xs.append(rdf[0, 1])
    rdf_y = np.mean(rdf_ys, axis=0)
    rdf_x = rdf_xs[0]

    ## Save the RDF
    import matplotlib.pyplot as plt

    plt.plot(rdf_x, rdf_y)
    plt.xlabel("r (Angstrom)")
    plt.ylabel("g(r)")
    plt.title(f"RDF {elements[0]}-{elements[1]}")
    plt.tight_layout()
    plt.savefig("rdf.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--md_traj", type=str, default="./md_results/md_traj.xyz")
    parser.add_argument("--n_frames", type=int, default=1000)
    parser.add_argument("--r_max", type=float, default=12.0)
    parser.add_argument("--n_bins", type=int, default=100)
    parser.add_argument("--elements", type=str, nargs="+", default=["O", "H"])
    args_dict = vars(parser.parse_args())
    main(args_dict)
