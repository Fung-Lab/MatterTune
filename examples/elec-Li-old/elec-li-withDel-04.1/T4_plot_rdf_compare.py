from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_AIMD_XYZ = Path(
    "/storage/lingyu/Electrolyte/get_all_trj_aimd/Li-metal/case3-Li-FSI-FEC/"
    "case1-case3-Li-FSI-FEC-1-13.0/"
    "case1-case3-Li-FSI-FEC-1-13.0_lambda_0.00.xyz"
)


def parse_indices(raw: str) -> np.ndarray:
    values = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one target index is required.")
    return np.asarray([int(value) for value in values], dtype=int)


def parse_species(raw: str) -> list[str]:
    values = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one neighbor species is required.")
    return values


def count_xyz_frames(path: Path, *, max_frames: int | None = None) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    natoms = int(first.strip())
    frame_lines = natoms + 2
    with path.open("r", encoding="utf-8") as handle:
        n_lines = sum(1 for _ in handle)
    if n_lines % frame_lines != 0:
        raise ValueError(f"{path} has {n_lines} lines, not divisible by frame size {frame_lines}.")
    n_frames = n_lines // frame_lines
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    return natoms, n_frames


def parse_comment_time_ps(comment: str) -> float | None:
    match = re.search(r"\btime_fs=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)", comment)
    if match:
        return float(match.group(1)) / 1000.0

    match = re.search(
        r"\btime\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
        comment,
    )
    if match:
        return float(match.group(1)) / 1000.0
    return None


def select_frame_indices(
    path: Path,
    *,
    max_frames: int | None,
    max_time_ps: float | None,
) -> tuple[int, list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    natoms = int(first.strip())

    selected: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        frame_idx = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            this_natoms = int(natoms_line.strip())
            if this_natoms != natoms:
                raise ValueError(f"{path} frame {frame_idx} has {this_natoms} atoms; expected {natoms}.")
            comment = handle.readline()
            if max_frames is not None and frame_idx >= max_frames:
                break

            keep = True
            if max_time_ps is not None:
                time_ps = parse_comment_time_ps(comment)
                if time_ps is None:
                    raise ValueError(
                        f"Could not parse frame time from {path} frame {frame_idx} comment: {comment!r}"
                    )
                keep = time_ps <= max_time_ps + 1.0e-12
                if not keep:
                    break

            if keep:
                selected.append(frame_idx)

            for _ in range(natoms):
                handle.readline()
            frame_idx += 1

    if not selected:
        raise ValueError(f"No frames selected from {path}.")
    return natoms, selected


def parse_position_line(line: str) -> tuple[str, np.ndarray]:
    fields = line.split()
    if len(fields) < 4:
        raise ValueError(f"Invalid XYZ atom line: {line!r}")
    return fields[0], np.asarray([float(fields[1]), float(fields[2]), float(fields[3])])


def minimum_image_distances(
    centers: np.ndarray,
    neighbors: np.ndarray,
    *,
    box_length: float,
    exclude_self: bool,
) -> np.ndarray:
    delta = neighbors[None, :, :] - centers[:, None, :]
    delta -= box_length * np.rint(delta / box_length)
    distances = np.linalg.norm(delta.reshape(-1, 3), axis=1)
    if exclude_self:
        distances = distances[distances > 1.0e-8]
    return distances


def initialize_index_groups(
    symbols: np.ndarray,
    *,
    target_indices: np.ndarray,
    center_species: str,
    neighbor_species: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    if np.any(target_indices < 0) or np.any(target_indices >= symbols.size):
        raise ValueError(f"Target indices out of range for {symbols.size} atoms: {target_indices}")
    groups = {
        "target": {"centers": target_indices},
        f"all_{center_species}": {"centers": np.flatnonzero(symbols == center_species)},
    }
    if groups[f"all_{center_species}"]["centers"].size == 0:
        raise ValueError(f"No center atoms of species {center_species} found.")
    for group in groups.values():
        for species in neighbor_species:
            indices = np.flatnonzero(symbols == species)
            if indices.size == 0:
                raise ValueError(f"No neighbor atoms of species {species} found.")
            group[species] = indices
    return groups


def accumulate_rdfs(
    path: Path,
    *,
    dataset_name: str,
    box_length: float,
    last_fraction: float,
    edges: np.ndarray,
    target_indices: np.ndarray,
    center_species: str,
    neighbor_species: list[str],
    max_frames: int | None,
    max_time_ps: float | None,
) -> dict[str, object]:
    natoms, selected_indices = select_frame_indices(
        path,
        max_frames=max_frames,
        max_time_ps=max_time_ps,
    )
    start_offset = math.floor((1.0 - last_fraction) * len(selected_indices))
    retained_indices = set(selected_indices[start_offset:])
    last_retained_index = selected_indices[-1]
    start_frame = selected_indices[start_offset]
    kept_frames = 0
    symbols_ref: np.ndarray | None = None
    index_groups: dict[str, dict[str, np.ndarray]] | None = None
    group_names = ("target", f"all_{center_species}")
    histograms = {
        (group_name, species): np.zeros(edges.size - 1, dtype=float)
        for group_name in group_names
        for species in neighbor_species
    }

    with path.open("r", encoding="utf-8") as handle:
        frame_idx = 0
        while frame_idx <= last_retained_index:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            this_natoms = int(natoms_line.strip())
            if this_natoms != natoms:
                raise ValueError(f"{path} frame {frame_idx} has {this_natoms} atoms; expected {natoms}.")
            _comment = handle.readline()

            if frame_idx not in retained_indices:
                for _ in range(natoms):
                    handle.readline()
                frame_idx += 1
                continue

            symbols: list[str] = []
            positions = np.empty((natoms, 3), dtype=float)
            for atom_idx in range(natoms):
                symbol, position = parse_position_line(handle.readline())
                symbols.append(symbol)
                positions[atom_idx] = position

            symbols_array = np.asarray(symbols)
            if symbols_ref is None:
                symbols_ref = symbols_array
                index_groups = initialize_index_groups(
                    symbols_ref,
                    target_indices=target_indices,
                    center_species=center_species,
                    neighbor_species=neighbor_species,
                )
            elif not np.array_equal(symbols_array, symbols_ref):
                raise ValueError(f"Species order changed in {path} frame {frame_idx}.")

            assert index_groups is not None
            for group_name in group_names:
                center_indices = index_groups[group_name]["centers"]
                centers = positions[center_indices]
                for species in neighbor_species:
                    neighbor_indices = index_groups[group_name][species]
                    neighbors = positions[neighbor_indices]
                    distances = minimum_image_distances(
                        centers,
                        neighbors,
                        box_length=box_length,
                        exclude_self=bool(set(center_indices).intersection(set(neighbor_indices))),
                    )
                    histograms[(group_name, species)] += np.histogram(distances, bins=edges)[0]
            kept_frames += 1
            frame_idx += 1

    if symbols_ref is None or index_groups is None:
        raise ValueError(f"No frames retained from {path}.")

    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rdf: dict[tuple[str, str], np.ndarray] = {}
    for key, hist in histograms.items():
        group_name, species = key
        n_centers = index_groups[group_name]["centers"].size
        n_neighbors = index_groups[group_name][species].size
        number_density = n_neighbors / box_length**3
        normalization = kept_frames * n_centers * number_density * shell_volumes
        rdf[key] = hist / normalization

    return {
        "dataset": dataset_name,
        "path": str(path),
        "natoms": natoms,
        "n_frames_total": len(selected_indices),
        "start_frame": start_frame,
        "n_frames_used": kept_frames,
        "rdf": rdf,
        "group_names": group_names,
        "neighbor_species": neighbor_species,
    }


def write_rdf_csv(
    aimd: dict[str, object],
    mlip: dict[str, object],
    r_mid: np.ndarray,
    path: Path,
) -> None:
    group_names = aimd["group_names"]
    neighbor_species = aimd["neighbor_species"]
    fields = ["r_A"]
    for group_name in group_names:
        for species in neighbor_species:
            fields.append(f"AIMD_{group_name}_{species}")
            fields.append(f"MLIP_{group_name}_{species}")

    aimd_rdf = aimd["rdf"]
    mlip_rdf = mlip["rdf"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bin_idx, radius in enumerate(r_mid):
            row: dict[str, float] = {"r_A": float(radius)}
            for group_name in group_names:
                for species in neighbor_species:
                    key = (group_name, species)
                    row[f"AIMD_{group_name}_{species}"] = float(aimd_rdf[key][bin_idx])
                    row[f"MLIP_{group_name}_{species}"] = float(mlip_rdf[key][bin_idx])
            writer.writerow(row)


def plot_rdfs(
    aimd: dict[str, object],
    mlip: dict[str, object],
    r_mid: np.ndarray,
    path: Path,
    *,
    r_max: float,
) -> None:
    group_names = list(aimd["group_names"])
    neighbor_species = list(aimd["neighbor_species"])
    aimd_rdf = aimd["rdf"]
    mlip_rdf = mlip["rdf"]

    def display_group(name: str) -> str:
        return "target Li" if name == "target" else name.replace("_", " ")

    fig, axes = plt.subplots(
        len(group_names),
        len(neighbor_species),
        figsize=(3.8 * len(neighbor_species), 3.2 * len(group_names)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for row_idx, group_name in enumerate(group_names):
        for col_idx, species in enumerate(neighbor_species):
            ax = axes[row_idx, col_idx]
            key = (group_name, species)
            ax.plot(r_mid, aimd_rdf[key], color="#1f2937", linewidth=1.7, label=str(aimd["dataset"]))
            ax.plot(r_mid, mlip_rdf[key], color="#2563eb", linewidth=1.7, label=str(mlip["dataset"]))
            ax.set_title(f"{display_group(group_name)}-{species}")
            ax.set_xlim(0.0, r_max)
            ax.grid(True, color="#d4d4d8", linewidth=0.7, alpha=0.7)
            if col_idx == 0:
                ax.set_ylabel("g(r)")
            if row_idx == len(group_names) - 1:
                ax.set_xlabel("r (A)")
            if row_idx == 0 and col_idx == len(neighbor_species) - 1:
                ax.legend(frameon=False)
    fig.suptitle(
        f"RDF comparison from last trajectory fraction "
        f"(AIMD {aimd['n_frames_used']} frames, MLIP {mlip['n_frames_used']} frames)"
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(aimd: dict[str, object], mlip: dict[str, object], path: Path, args: argparse.Namespace) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"AIMD trajectory: {aimd['path']}\n")
        handle.write(f"MLIP trajectory: {mlip['path']}\n")
        handle.write(f"cell: cubic {args.cell_length_a:.6f} A\n")
        handle.write(f"rmax: {args.r_max_a:.3f} A\n")
        handle.write(f"dr: {args.dr_a:.3f} A\n")
        handle.write(f"last_fraction: {args.last_fraction:.3f}\n")
        handle.write(f"max_time_ps: {args.max_time_ps}\n")
        handle.write(f"target_indices: {args.target_indices}\n")
        handle.write(f"center_species: {args.center_species}\n")
        handle.write(f"neighbor_species: {args.neighbor_species}\n\n")
        for data in (aimd, mlip):
            handle.write(f"[{data['dataset']}]\n")
            handle.write(f"total frames considered: {data['n_frames_total']}\n")
            handle.write(f"start frame: {data['start_frame']}\n")
            handle.write(f"frames used: {data['n_frames_used']}\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AIMD and MLIP-MD atom RDFs over the last R fraction."
    )
    parser.add_argument("--aimd-xyz", type=Path, default=DEFAULT_AIMD_XYZ)
    parser.add_argument("--mlip-xyz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--aimd-label", default="AIMD")
    parser.add_argument("--mlip-label", default="MLIP-MD")
    parser.add_argument("--target-indices", default="0")
    parser.add_argument("--center-species", default="Li")
    parser.add_argument("--neighbor-species", default="C,H,F,N,O,S")
    parser.add_argument("--cell-length-a", type=float, default=15.569)
    parser.add_argument("--last-fraction", type=float, default=0.20)
    parser.add_argument(
        "--max-time-ps",
        type=float,
        default=None,
        help="Keep only initial frames with parsed time <= this value before applying --last-fraction.",
    )
    parser.add_argument("--r-max-a", type=float, default=7.5)
    parser.add_argument("--dr-a", type=float, default=0.05)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--prefix", default="T4_rdf_compare")
    args = parser.parse_args()
    if not args.aimd_xyz.is_file():
        raise FileNotFoundError(args.aimd_xyz)
    if not args.mlip_xyz.is_file():
        raise FileNotFoundError(args.mlip_xyz)
    if not 0.0 < args.last_fraction <= 1.0:
        parser.error("--last-fraction must be in (0, 1].")
    if args.r_max_a <= 0.0 or args.dr_a <= 0.0:
        parser.error("--r-max-a and --dr-a must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive.")
    if args.max_time_ps is not None and args.max_time_ps <= 0.0:
        parser.error("--max-time-ps must be positive.")
    return args


def main() -> None:
    args = parse_args()
    target_indices = parse_indices(args.target_indices)
    neighbor_species = parse_species(args.neighbor_species)
    out_dir = args.out_dir or args.mlip_xyz.parent / "T4_rdf_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = np.arange(0.0, args.r_max_a + args.dr_a, args.dr_a)
    r_mid = 0.5 * (edges[:-1] + edges[1:])

    aimd = accumulate_rdfs(
        args.aimd_xyz,
        dataset_name=args.aimd_label,
        box_length=args.cell_length_a,
        last_fraction=args.last_fraction,
        edges=edges,
        target_indices=target_indices,
        center_species=args.center_species,
        neighbor_species=neighbor_species,
        max_frames=args.max_frames,
        max_time_ps=args.max_time_ps,
    )
    mlip = accumulate_rdfs(
        args.mlip_xyz,
        dataset_name=args.mlip_label,
        box_length=args.cell_length_a,
        last_fraction=args.last_fraction,
        edges=edges,
        target_indices=target_indices,
        center_species=args.center_species,
        neighbor_species=neighbor_species,
        max_frames=args.max_frames,
        max_time_ps=args.max_time_ps,
    )
    plot_path = out_dir / f"{args.prefix}.png"
    csv_path = out_dir / f"{args.prefix}.csv"
    summary_path = out_dir / f"{args.prefix}.txt"
    plot_rdfs(aimd, mlip, r_mid, plot_path, r_max=args.r_max_a)
    write_rdf_csv(aimd, mlip, r_mid, csv_path)
    write_summary(aimd, mlip, summary_path, args)
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
