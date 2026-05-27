from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import T5_plot_mol_com_rdf_compare as t5
import T6_plot_mol_com_rdf_windows as t6


def plot_pair_window_overlay(
    aimd: dict[str, object],
    mlip: dict[str, object],
    r_mid_nm: np.ndarray,
    key: tuple[str, str],
    path: Path,
    *,
    r_max_nm: float,
) -> None:
    windows = aimd["windows"]
    center_name, neighbor_name = key
    neighbor_label = t5.reference_mode_label(neighbor_name, str(aimd["neighbor_center"]))
    title = f"{t6.display_center(center_name)}-{neighbor_label} molecule COM RDF by time window"
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(windows)))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, data in zip(axes, (aimd, mlip), strict=True):
        plotted = False
        for window_idx, (start_ps, end_ps) in enumerate(windows):
            values = data["rdf_by_window"][window_idx][key]
            frames = int(data["frames_used"][window_idx])
            if not t6.has_curve(values):
                continue
            ax.plot(
                r_mid_nm,
                values,
                color=colors[window_idx],
                linewidth=1.6,
                label=f"{start_ps:g}-{end_ps:g} ps ({frames})",
            )
            plotted = True
        if not plotted:
            ax.text(0.05, 0.90, "no frames", transform=ax.transAxes, color="#6b7280", fontsize=10)
        ax.set_title(str(data["dataset"]))
        ax.set_xlim(0.0, r_max_nm)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, color="#d4d4d8", linewidth=0.7, alpha=0.7)
        ax.set_xlabel("r (nm)")
    axes[0].set_ylabel("g(r)")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot molecule-COM RDF time-window overlays with AIMD windows in one subplot "
            "and MLIP-MD windows in another."
        )
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
    parser.add_argument("--prefix", default="T7_mol_com_rdf_window_overlay")
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

    out_dir = args.out_dir or args.mlip_xyz.parent / "T7_mol_com_rdf_window_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = t6.make_windows(args.total_time_ps, args.window_ps)
    edges_a = np.arange(0.0, args.r_max_nm * 10.0 + args.dr_nm * 10.0, args.dr_nm * 10.0)
    r_mid_nm = 0.5 * (edges_a[:-1] + edges_a[1:]) / 10.0

    aimd = t6.accumulate_windowed_mol_com_rdfs(
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
    mlip = t6.accumulate_windowed_mol_com_rdfs(
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
        suffix = "" if len(pair_keys) == 1 else f"_{t6.safe_label(center_name)}_{t6.safe_label(neighbor_name)}"
        plot_path = out_dir / f"{args.prefix}{suffix}.png"
        csv_path = out_dir / f"{args.prefix}{suffix}.csv"
        plot_pair_window_overlay(aimd, mlip, r_mid_nm, key, plot_path, r_max_nm=args.r_max_nm)
        t6.write_pair_csv(aimd, mlip, r_mid_nm, key, csv_path)
        written.extend([plot_path, csv_path])

    summary_path = out_dir / f"{args.prefix}.txt"
    t6.write_summary(aimd, mlip, summary_path, args, box_length=box_length)
    written.append(summary_path)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
