from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import T5_plot_mol_com_rdf_compare as t5


def make_windows(total_time_ps: float, window_ps: float) -> list[tuple[float, float]]:
    n_windows = int(math.ceil(total_time_ps / window_ps))
    return [
        (idx * window_ps, min((idx + 1) * window_ps, total_time_ps))
        for idx in range(n_windows)
    ]


def find_window_index(
    time_ps: float,
    windows: list[tuple[float, float]],
    *,
    tol: float = 1.0e-12,
) -> int | None:
    for idx, (start_ps, end_ps) in enumerate(windows):
        is_last = idx == len(windows) - 1
        if time_ps >= start_ps - tol and (time_ps < end_ps - tol or (is_last and time_ps <= end_ps + tol)):
            return idx
    return None


def safe_label(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def display_center(name: str) -> str:
    return name[4:] if name.startswith("all_") else name


def read_frame_time_ps(comment: str, *, frame_idx: int, dt_fs: float | None, path: Path) -> float:
    time_ps = t5.parse_comment_time_ps(comment)
    if time_ps is not None:
        return time_ps
    if dt_fs is not None:
        return frame_idx * dt_fs / 1000.0
    raise ValueError(
        f"Could not parse frame time from {path} frame {frame_idx}; "
        "provide --aimd-dt-fs or --mlip-dt-fs as a fallback."
    )


def accumulate_windowed_mol_com_rdfs(
    path: Path,
    *,
    dataset_name: str,
    molecules: list[t5.Molecule],
    topology_natoms: int,
    box_length: float,
    windows: list[tuple[float, float]],
    dt_fs: float | None,
    edges_a: np.ndarray,
    center_resnames: list[str],
    neighbor_resnames: list[str],
    target_indices: np.ndarray,
    center_mode: str,
    neighbor_center: str,
) -> dict[str, object]:
    center_groups, neighbor_groups = t5.build_groups(
        molecules,
        center_resnames=center_resnames,
        neighbor_resnames=neighbor_resnames,
        target_indices=target_indices,
        center_mode=center_mode,
    )
    pair_keys = [
        (center_name, neighbor_name)
        for center_name in center_groups
        for neighbor_name in neighbor_groups
    ]
    histograms = [
        {key: np.zeros(edges_a.size - 1, dtype=float) for key in pair_keys}
        for _ in windows
    ]
    frames_used = np.zeros(len(windows), dtype=int)
    first_frame = [None for _ in windows]
    last_frame = [None for _ in windows]
    neighbor_ids_needed = np.unique(np.concatenate(list(neighbor_groups.values())))

    with path.open("r", encoding="utf-8") as handle:
        frame_idx = 0
        natoms: int | None = None
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            this_natoms = int(natoms_line.strip())
            if natoms is None:
                natoms = this_natoms
                if natoms != topology_natoms:
                    raise ValueError(f"{path} has {natoms} atoms, but topology has {topology_natoms} atoms.")
            elif this_natoms != natoms:
                raise ValueError(f"{path} frame {frame_idx} has {this_natoms} atoms; expected {natoms}.")

            comment = handle.readline()
            time_ps = read_frame_time_ps(comment, frame_idx=frame_idx, dt_fs=dt_fs, path=path)
            if time_ps > windows[-1][1] + 1.0e-12:
                break
            window_idx = find_window_index(time_ps, windows)
            if window_idx is None:
                for _ in range(natoms):
                    handle.readline()
                frame_idx += 1
                continue

            positions = np.empty((natoms, 3), dtype=float)
            for atom_idx in range(natoms):
                positions[atom_idx] = t5.parse_position_line(handle.readline())

            center_positions = t5.molecule_centers(
                positions,
                molecules,
                box_length=box_length,
                reference_mode="mol_com",
            )
            neighbor_positions = t5.molecule_centers(
                positions,
                molecules,
                box_length=box_length,
                reference_mode=neighbor_center,
                required_mol_ids=neighbor_ids_needed,
            )
            for center_name, center_ids in center_groups.items():
                centers = center_positions[center_ids]
                for neighbor_name, neighbor_ids in neighbor_groups.items():
                    neighbors = neighbor_positions[neighbor_ids]
                    distances = t5.minimum_image_pair_distances(
                        centers,
                        neighbors,
                        center_ids=center_ids,
                        neighbor_ids=neighbor_ids,
                        box_length=box_length,
                    )
                    histograms[window_idx][(center_name, neighbor_name)] += np.histogram(distances, bins=edges_a)[0]

            frames_used[window_idx] += 1
            if first_frame[window_idx] is None:
                first_frame[window_idx] = frame_idx
            last_frame[window_idx] = frame_idx
            frame_idx += 1

    if natoms is None:
        raise ValueError(f"No frames found in {path}.")

    shell_volumes = (4.0 / 3.0) * np.pi * (edges_a[1:] ** 3 - edges_a[:-1] ** 3)
    rdf_by_window: list[dict[tuple[str, str], np.ndarray]] = []
    for window_idx, hist_by_pair in enumerate(histograms):
        window_rdf: dict[tuple[str, str], np.ndarray] = {}
        for key, hist in hist_by_pair.items():
            center_name, neighbor_name = key
            center_ids = center_groups[center_name]
            neighbor_ids = neighbor_groups[neighbor_name]
            valid_pairs = center_ids.size * neighbor_ids.size - len(set(center_ids).intersection(set(neighbor_ids)))
            if valid_pairs <= 0:
                raise ValueError(f"No valid molecule pairs for {center_name}-{neighbor_name}.")
            if frames_used[window_idx] == 0:
                window_rdf[key] = np.full(edges_a.size - 1, np.nan, dtype=float)
                continue
            normalization = frames_used[window_idx] * valid_pairs * shell_volumes / box_length**3
            window_rdf[key] = hist / normalization
        rdf_by_window.append(window_rdf)

    return {
        "dataset": dataset_name,
        "path": str(path),
        "natoms": natoms,
        "windows": windows,
        "frames_used": frames_used,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "rdf_by_window": rdf_by_window,
        "center_groups": center_groups,
        "neighbor_groups": neighbor_groups,
        "neighbor_center": t5.normalize_reference_mode(neighbor_center),
    }


def has_curve(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).any())


def write_pair_csv(
    aimd: dict[str, object],
    mlip: dict[str, object],
    r_mid_nm: np.ndarray,
    key: tuple[str, str],
    path: Path,
) -> None:
    windows = aimd["windows"]
    fieldnames = ["r_nm"]
    for start_ps, end_ps in windows:
        label = f"{start_ps:g}_{end_ps:g}ps"
        fieldnames.append(f"AIMD_{label}")
        fieldnames.append(f"MLIP_{label}")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bin_idx, radius in enumerate(r_mid_nm):
            row: dict[str, float | str] = {"r_nm": float(radius)}
            for window_idx, (start_ps, end_ps) in enumerate(windows):
                label = f"{start_ps:g}_{end_ps:g}ps"
                aimd_value = aimd["rdf_by_window"][window_idx][key][bin_idx]
                mlip_value = mlip["rdf_by_window"][window_idx][key][bin_idx]
                row[f"AIMD_{label}"] = "" if not np.isfinite(aimd_value) else float(aimd_value)
                row[f"MLIP_{label}"] = "" if not np.isfinite(mlip_value) else float(mlip_value)
            writer.writerow(row)


def plot_pair_windows(
    aimd: dict[str, object],
    mlip: dict[str, object],
    r_mid_nm: np.ndarray,
    key: tuple[str, str],
    path: Path,
    *,
    r_max_nm: float,
) -> None:
    windows = aimd["windows"]
    n_windows = len(windows)
    n_cols = min(n_windows, 5)
    n_rows = int(math.ceil(n_windows / n_cols))
    center_name, neighbor_name = key
    neighbor_label = t5.reference_mode_label(neighbor_name, str(aimd["neighbor_center"]))
    title = f"{display_center(center_name)}-{neighbor_label} molecule COM RDF"

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.3 * n_cols, 3.0 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for window_idx, ax in enumerate(flat_axes):
        if window_idx >= n_windows:
            ax.axis("off")
            continue
        start_ps, end_ps = windows[window_idx]
        aimd_values = aimd["rdf_by_window"][window_idx][key]
        mlip_values = mlip["rdf_by_window"][window_idx][key]
        aimd_frames = int(aimd["frames_used"][window_idx])
        mlip_frames = int(mlip["frames_used"][window_idx])

        if has_curve(aimd_values):
            ax.plot(r_mid_nm, aimd_values, color="#1f2937", linewidth=1.7, label=str(aimd["dataset"]))
        else:
            ax.text(0.04, 0.90, "AIMD: no frames", transform=ax.transAxes, color="#6b7280", fontsize=9)
        if has_curve(mlip_values):
            ax.plot(r_mid_nm, mlip_values, color="#2563eb", linewidth=1.7, label=str(mlip["dataset"]))
        else:
            ax.text(0.04, 0.80, "MLIP: no frames", transform=ax.transAxes, color="#6b7280", fontsize=9)

        ax.set_title(f"{start_ps:g}-{end_ps:g} ps\nAIMD {aimd_frames}, MLIP {mlip_frames} frames")
        ax.set_xlim(0.0, r_max_nm)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, color="#d4d4d8", linewidth=0.7, alpha=0.7)
        if window_idx % n_cols == 0:
            ax.set_ylabel("g(r)")
        if window_idx // n_cols == n_rows - 1:
            ax.set_xlabel("r (nm)")
        if window_idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False)

    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(
    aimd: dict[str, object],
    mlip: dict[str, object],
    path: Path,
    args: argparse.Namespace,
    *,
    box_length: float,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"AIMD trajectory: {aimd['path']}\n")
        handle.write(f"MLIP trajectory: {mlip['path']}\n")
        handle.write(f"topology PDB: {args.top_pdb}\n")
        handle.write(f"cell: cubic {box_length:.6f} A\n")
        handle.write(f"total_time_ps: {args.total_time_ps:.6f}\n")
        handle.write(f"window_ps: {args.window_ps:.6f}\n")
        handle.write(f"rmax: {args.r_max_nm:.4f} nm\n")
        handle.write(f"dr: {args.dr_nm:.4f} nm\n")
        handle.write(f"target_indices: {args.target_indices}\n")
        handle.write(f"center_resnames: {args.center_resnames}\n")
        handle.write(f"neighbor_resnames: {args.neighbor_resnames}\n")
        handle.write(f"neighbor_center: {args.neighbor_center}\n\n")
        for idx, (start_ps, end_ps) in enumerate(aimd["windows"]):
            handle.write(f"[{start_ps:g}-{end_ps:g} ps]\n")
            for data in (aimd, mlip):
                handle.write(
                    f"{data['dataset']}: frames={int(data['frames_used'][idx])}, "
                    f"first_frame={data['first_frame'][idx]}, last_frame={data['last_frame'][idx]}\n"
                )
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AIMD and MLIP-MD molecule-COM RDFs in fixed time windows."
    )
    parser.add_argument("--aimd-xyz", type=Path, default=t5.DEFAULT_AIMD_XYZ)
    parser.add_argument("--mlip-xyz", type=Path, required=True)
    parser.add_argument("--top-pdb", type=Path, default=t5.DEFAULT_TOP_PDB)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--aimd-label", default="AIMD")
    parser.add_argument("--mlip-label", default="MLIP-MD")
    parser.add_argument("--target-indices", default="0")
    parser.add_argument("--center-resnames", default="Li")
    parser.add_argument("--neighbor-resnames", default="FEC")
    parser.add_argument("--center-mode", choices=("target", "all", "both"), default="all")
    parser.add_argument(
        "--neighbor-center",
        default="mol_com",
        help="Neighbor reference point: mol_com, mol_cog, or atom:<PDB atom name> such as atom:N.",
    )
    parser.add_argument("--cell-length-a", type=float, default=None)
    parser.add_argument("--total-time-ps", type=float, default=50.0)
    parser.add_argument("--window-ps", type=float, default=10.0)
    parser.add_argument("--aimd-dt-fs", type=float, default=None)
    parser.add_argument("--mlip-dt-fs", type=float, default=None)
    parser.add_argument("--r-max-nm", type=float, default=0.75)
    parser.add_argument("--dr-nm", type=float, default=0.01)
    parser.add_argument("--prefix", default="T6_mol_com_rdf_windows")
    args = parser.parse_args()

    if not args.aimd_xyz.is_file():
        raise FileNotFoundError(args.aimd_xyz)
    if not args.mlip_xyz.is_file():
        raise FileNotFoundError(args.mlip_xyz)
    if not args.top_pdb.is_file():
        raise FileNotFoundError(args.top_pdb)
    if args.total_time_ps <= 0.0 or args.window_ps <= 0.0:
        parser.error("--total-time-ps and --window-ps must be positive.")
    if args.r_max_nm <= 0.0 or args.dr_nm <= 0.0:
        parser.error("--r-max-nm and --dr-nm must be positive.")
    if args.cell_length_a is not None and args.cell_length_a <= 0.0:
        parser.error("--cell-length-a must be positive.")
    for name in ("aimd_dt_fs", "mlip_dt_fs"):
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    try:
        args.neighbor_center = t5.normalize_reference_mode(args.neighbor_center)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    center_resnames = t5.parse_csv_list(args.center_resnames)
    neighbor_resnames = t5.parse_csv_list(args.neighbor_resnames)
    target_indices = t5.parse_indices(args.target_indices)
    molecules, topology_natoms, topology_box_length = t5.load_topology(args.top_pdb)
    box_length = args.cell_length_a if args.cell_length_a is not None else topology_box_length
    if box_length is None:
        raise ValueError("No cell length provided and no cubic CRYST1 record found in top PDB.")

    out_dir = args.out_dir or args.mlip_xyz.parent / "T6_mol_com_rdf_windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = make_windows(args.total_time_ps, args.window_ps)
    edges_a = np.arange(0.0, args.r_max_nm * 10.0 + args.dr_nm * 10.0, args.dr_nm * 10.0)
    r_mid_nm = 0.5 * (edges_a[:-1] + edges_a[1:]) / 10.0

    aimd = accumulate_windowed_mol_com_rdfs(
        args.aimd_xyz,
        dataset_name=args.aimd_label,
        molecules=molecules,
        topology_natoms=topology_natoms,
        box_length=box_length,
        windows=windows,
        dt_fs=args.aimd_dt_fs,
        edges_a=edges_a,
        center_resnames=center_resnames,
        neighbor_resnames=neighbor_resnames,
        target_indices=target_indices,
        center_mode=args.center_mode,
        neighbor_center=args.neighbor_center,
    )
    mlip = accumulate_windowed_mol_com_rdfs(
        args.mlip_xyz,
        dataset_name=args.mlip_label,
        molecules=molecules,
        topology_natoms=topology_natoms,
        box_length=box_length,
        windows=windows,
        dt_fs=args.mlip_dt_fs,
        edges_a=edges_a,
        center_resnames=center_resnames,
        neighbor_resnames=neighbor_resnames,
        target_indices=target_indices,
        center_mode=args.center_mode,
        neighbor_center=args.neighbor_center,
    )

    center_groups = list(aimd["center_groups"])
    neighbor_groups = list(aimd["neighbor_groups"])
    pair_keys = [(center_name, neighbor_name) for center_name in center_groups for neighbor_name in neighbor_groups]
    written: list[Path] = []
    for key in pair_keys:
        center_name, neighbor_name = key
        suffix = "" if len(pair_keys) == 1 else f"_{safe_label(center_name)}_{safe_label(neighbor_name)}"
        plot_path = out_dir / f"{args.prefix}{suffix}.png"
        csv_path = out_dir / f"{args.prefix}{suffix}.csv"
        plot_pair_windows(aimd, mlip, r_mid_nm, key, plot_path, r_max_nm=args.r_max_nm)
        write_pair_csv(aimd, mlip, r_mid_nm, key, csv_path)
        written.extend([plot_path, csv_path])

    summary_path = out_dir / f"{args.prefix}.txt"
    write_summary(aimd, mlip, summary_path, args, box_length=box_length)
    written.append(summary_path)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
