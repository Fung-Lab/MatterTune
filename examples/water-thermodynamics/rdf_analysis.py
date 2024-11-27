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
    n_bins = int(args_dict["r_max"] / args_dict["r_step"])

    elements = ["O", "O"]
    if not args_dict["load"]:
        rdf_ys = []
        rdf_xs = []
        for atoms in md_traj:
            rdf = rdf_compute(atoms, r_max, n_bins, elements)
            rdf_ys.append(rdf[0, 0])
            rdf_xs.append(rdf[0, 1])
        rdf_y = np.mean(rdf_ys, axis=0)
        rdf_x = rdf_xs[0]
        np.savez(
            f"./plots/rdf-{elements[0]}-{elements[1]}.npz", rdf_x=rdf_x, rdf_y=rdf_y
        )
    data = np.load(f"./plots/rdf-{elements[0]}-{elements[1]}.npz")
    rdf_x = data["rdf_x"]
    rdf_y = data["rdf_y"]
    sigma = 1.0
    smoothed_rdf_y = gaussian_filter1d(rdf_y, sigma=sigma)
    plt.figure(figsize=(6, 3))
    plt.plot(rdf_x, smoothed_rdf_y, color="green", linewidth=2)
    plt.ylim(0, 3)
    plt.xlim(0.5, 6.2)
    plt.yticks([0, 1, 2, 3])
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.tight_layout(pad=0)
    if args_dict["transparent"]:
        plt.axis("off")
        plt.gca().set_facecolor("none")
        plt.gcf().set_facecolor("none")
    plt.savefig(
        f"./plots/rdf-{elements[0]}-{elements[1]}.png",
        dpi=300,
        transparent=args_dict["transparent"],
    )

    elements = ["O", "H"]
    rdf_ys = []
    rdf_xs = []
    if not args_dict["load"]:
        for atoms in md_traj:
            rdf = rdf_compute(atoms, r_max, n_bins, elements)
            rdf_ys.append(rdf[0, 0])
            rdf_xs.append(rdf[0, 1])
        rdf_y = np.mean(rdf_ys, axis=0)
        rdf_x = rdf_xs[0]
        np.savez(
            f"./plots/rdf-{elements[0]}-{elements[1]}.npz", rdf_x=rdf_x, rdf_y=rdf_y
        )
    data = np.load(f"./plots/rdf-{elements[0]}-{elements[1]}.npz")
    rdf_x = data["rdf_x"]
    rdf_y = data["rdf_y"]
    sigma = 2.0
    smoothed_rdf_y = gaussian_filter1d(rdf_y, sigma=sigma)
    plt.figure(figsize=(6, 3))
    plt.plot(rdf_x, smoothed_rdf_y, color="green", linewidth=2)
    plt.ylim(0, 2.5)
    plt.xlim(0.5, 6.2)
    plt.yticks([0, 1, 2])
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.tight_layout(pad=0)
    if args_dict["transparent"]:
        plt.axis("off")
        plt.gca().set_facecolor("none")
        plt.gcf().set_facecolor("none")
    plt.savefig(
        f"./plots/rdf-{elements[0]}-{elements[1]}.png",
        dpi=300,
        transparent=args_dict["transparent"],
    )

    elements = ["H", "H"]
    rdf_ys = []
    rdf_xs = []
    if not args_dict["load"]:
        for atoms in md_traj:
            rdf = rdf_compute(atoms, r_max, n_bins, elements)
            rdf_ys.append(rdf[0, 0])
            rdf_xs.append(rdf[0, 1])
        rdf_y = np.mean(rdf_ys, axis=0)
        rdf_x = rdf_xs[0]
        np.savez(
            f"./plots/rdf-{elements[0]}-{elements[1]}.npz", rdf_x=rdf_x, rdf_y=rdf_y
        )
    data = np.load(f"./plots/rdf-{elements[0]}-{elements[1]}.npz")
    rdf_x = data["rdf_x"]
    rdf_y = data["rdf_y"]
    sigma = 1.0
    smoothed_rdf_y = gaussian_filter1d(rdf_y, sigma=sigma)
    plt.figure(figsize=(6, 3))
    plt.plot(rdf_x, smoothed_rdf_y, color="green", linewidth=2)
    plt.ylim(0, 2.2)
    plt.xlim(0.5, 6.2)
    plt.yticks([0, 1, 2])
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.tight_layout(pad=0)
    if args_dict["transparent"]:
        plt.axis("off")
        plt.gca().set_facecolor("none")
        plt.gcf().set_facecolor("none")
    plt.savefig(
        f"./plots/rdf-{elements[0]}-{elements[1]}.png",
        dpi=300,
        transparent=args_dict["transparent"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--md_traj", type=str, default="./md_results/md_traj_fric0.05_orb-best-0.03.xyz"
    )
    parser.add_argument("--n_frames", type=int, default=2000)
    parser.add_argument("--r_max", type=float, default=6.0)
    parser.add_argument("--r_step", type=float, default=0.06)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--load", action="store_true")
    args_dict = vars(parser.parse_args())
    main(args_dict)
