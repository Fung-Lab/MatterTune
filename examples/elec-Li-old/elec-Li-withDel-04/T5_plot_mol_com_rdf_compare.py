from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_AIMD_XYZ = Path(
    "/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/get_all_trj_aimd/Li-metal/"
    "case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/"
    "case1-case3-Li-FSI-FEC-1-13.0_lambda_0.00.xyz"
)
DEFAULT_TOP_PDB = Path(
    "/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/get_all_trj_aimd/Li-metal/"
    "case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/top.pdb"
)

MASS_BY_ELEMENT = {
    "H": 1.00794,
    "Li": 6.941,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "S": 32.065,
    "P": 30.973762,
    "B": 10.811,
    "Na": 22.98976928,
    "K": 39.0983,
    "Cl": 35.453,
}


@dataclass(frozen=True)
class Molecule:
    mol_id: int
    resname: str
    resid: str
    atom_indices: np.ndarray
    masses: np.ndarray
    atom_names: tuple[str, ...]


def parse_csv_list(raw: str) -> list[str]:
    values = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not values:
        raise ValueError(f"Expected at least one value in {raw!r}.")
    return values


def parse_indices(raw: str) -> np.ndarray:
    values = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one target index is required.")
    return np.asarray([int(value) for value in values], dtype=int)


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


def infer_element(atom_name: str, element_field: str) -> str:
    raw = element_field.strip()
    if raw:
        candidate = raw[0].upper() + raw[1:].lower()
        if candidate in MASS_BY_ELEMENT:
            return candidate

    letters = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if not letters:
        raise ValueError(f"Could not infer element from atom name {atom_name!r}.")
    if len(letters) >= 2:
        candidate = letters[:2][0].upper() + letters[:2][1].lower()
        if candidate in MASS_BY_ELEMENT:
            return candidate
    candidate = letters[0].upper()
    if candidate in MASS_BY_ELEMENT:
        return candidate
    raise ValueError(f"Unsupported element inferred from atom name {atom_name!r}: {candidate}")


def parse_cryst1_box_length(line: str) -> float | None:
    try:
        a = float(line[6:15])
        b = float(line[15:24])
        c = float(line[24:33])
    except ValueError:
        return None
    if not (abs(a - b) < 1.0e-6 and abs(a - c) < 1.0e-6):
        raise ValueError(f"Only cubic boxes are supported by this script; got CRYST1 {a}, {b}, {c}.")
    return a


def load_topology(path: Path) -> tuple[list[Molecule], int, float | None]:
    grouped: dict[tuple[str, str, str, str], list[tuple[int, float, str]]] = {}
    order: list[tuple[str, str, str, str]] = []
    atom_index = 0
    box_length: float | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("CRYST1"):
                box_length = parse_cryst1_box_length(line)
                continue
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21:22].strip()
            resid = line[22:26].strip()
            insertion = line[26:27].strip()
            element = infer_element(atom_name, line[76:78] if len(line) >= 78 else "")
            key = (resname, chain, resid, insertion)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append((atom_index, MASS_BY_ELEMENT[element], atom_name))
            atom_index += 1

    if atom_index == 0:
        raise ValueError(f"No ATOM/HETATM records found in {path}.")

    molecules: list[Molecule] = []
    for mol_id, key in enumerate(order):
        resname, _chain, resid, insertion = key
        members = grouped[key]
        molecules.append(
            Molecule(
                mol_id=mol_id,
                resname=resname,
                resid=f"{resid}{insertion}",
                atom_indices=np.asarray([idx for idx, _mass, _name in members], dtype=int),
                masses=np.asarray([mass for _idx, mass, _name in members], dtype=float),
                atom_names=tuple(name for _idx, _mass, name in members),
            )
        )
    return molecules, atom_index, box_length


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


def parse_position_line(line: str) -> np.ndarray:
    fields = line.split()
    if len(fields) < 4:
        raise ValueError(f"Invalid XYZ atom line: {line!r}")
    return np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=float)


def normalize_reference_mode(raw: str) -> str:
    mode = raw.strip()
    if mode in {"mol_com", "mol_cog"}:
        return mode
    if mode.startswith("atom:") and mode.split(":", 1)[1].strip():
        return f"atom:{mode.split(':', 1)[1].strip()}"
    raise ValueError(
        f"Unsupported molecule reference mode {raw!r}. "
        "Use mol_com, mol_cog, or atom:<PDB atom name> such as atom:N."
    )


def reference_mode_label(resname: str, mode: str) -> str:
    if mode == "mol_com":
        return resname
    if mode == "mol_cog":
        return f"{resname}(COG)"
    if mode.startswith("atom:"):
        return f"{resname}({mode.split(':', 1)[1]})"
    return f"{resname}({mode})"


def molecule_centers(
    positions: np.ndarray,
    molecules: list[Molecule],
    *,
    box_length: float,
    reference_mode: str,
    required_mol_ids: np.ndarray | None = None,
) -> np.ndarray:
    reference_mode = normalize_reference_mode(reference_mode)
    required = None if required_mol_ids is None else set(int(idx) for idx in required_mol_ids)
    centers = np.empty((len(molecules), 3), dtype=float)
    for molecule in molecules:
        if required is not None and molecule.mol_id not in required:
            centers[molecule.mol_id] = np.nan
            continue
        mol_positions = positions[molecule.atom_indices]
        if reference_mode.startswith("atom:"):
            atom_name = reference_mode.split(":", 1)[1]
            local_matches = [
                idx
                for idx, name in enumerate(molecule.atom_names)
                if name == atom_name or name.upper() == atom_name.upper()
            ]
            if not local_matches:
                raise ValueError(
                    f"Molecule {molecule.resname}{molecule.resid} has no PDB atom name {atom_name!r}."
                )
            matched_positions = mol_positions[np.asarray(local_matches, dtype=int)]
            anchor = matched_positions[0]
            relative = matched_positions - anchor
            relative -= box_length * np.rint(relative / box_length)
            centers[molecule.mol_id] = (anchor + relative).mean(axis=0) % box_length
            continue

        anchor = mol_positions[0]
        relative = mol_positions - anchor
        relative -= box_length * np.rint(relative / box_length)
        unwrapped = anchor + relative
        if reference_mode == "mol_com":
            center = np.average(unwrapped, axis=0, weights=molecule.masses)
        else:
            center = unwrapped.mean(axis=0)
        centers[molecule.mol_id] = center % box_length
    return centers


def minimum_image_pair_distances(
    centers: np.ndarray,
    neighbors: np.ndarray,
    *,
    center_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    box_length: float,
) -> np.ndarray:
    delta = neighbors[None, :, :] - centers[:, None, :]
    delta -= box_length * np.rint(delta / box_length)
    distances = np.linalg.norm(delta, axis=2)
    valid = center_ids[:, None] != neighbor_ids[None, :]
    return distances[valid]


def mol_ids_by_resname(molecules: list[Molecule], resnames: list[str]) -> np.ndarray:
    wanted = set(resnames)
    ids = [molecule.mol_id for molecule in molecules if molecule.resname in wanted]
    if not ids:
        raise ValueError(f"No molecules found for resnames {','.join(resnames)}.")
    return np.asarray(ids, dtype=int)


def target_mol_ids(
    molecules: list[Molecule],
    *,
    target_indices: np.ndarray,
    center_resnames: list[str],
) -> np.ndarray:
    wanted_resnames = set(center_resnames)
    wanted_atoms = set(int(index) for index in target_indices)
    ids: list[int] = []
    for molecule in molecules:
        if molecule.resname not in wanted_resnames:
            continue
        if any(int(index) in wanted_atoms for index in molecule.atom_indices):
            ids.append(molecule.mol_id)
    if not ids:
        raise ValueError(
            "No center molecules contain target atom indices "
            f"{','.join(str(int(i)) for i in target_indices)}."
        )
    return np.asarray(ids, dtype=int)


def build_groups(
    molecules: list[Molecule],
    *,
    center_resnames: list[str],
    neighbor_resnames: list[str],
    target_indices: np.ndarray,
    center_mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    center_groups: dict[str, np.ndarray] = {}
    center_label = "_".join(center_resnames)
    if center_mode in {"target", "both"}:
        center_groups["target"] = target_mol_ids(
            molecules,
            target_indices=target_indices,
            center_resnames=center_resnames,
        )
    if center_mode in {"all", "both"}:
        center_groups[f"all_{center_label}"] = mol_ids_by_resname(molecules, center_resnames)
    neighbor_groups = {
        resname: mol_ids_by_resname(molecules, [resname])
        for resname in neighbor_resnames
    }
    return center_groups, neighbor_groups


def accumulate_mol_com_rdfs(
    path: Path,
    *,
    dataset_name: str,
    molecules: list[Molecule],
    topology_natoms: int,
    box_length: float,
    last_fraction: float,
    edges_a: np.ndarray,
    center_resnames: list[str],
    neighbor_resnames: list[str],
    target_indices: np.ndarray,
    center_mode: str,
    neighbor_center: str,
    max_frames: int | None,
    max_time_ps: float | None,
) -> dict[str, object]:
    natoms, selected_indices = select_frame_indices(
        path,
        max_frames=max_frames,
        max_time_ps=max_time_ps,
    )
    if natoms != topology_natoms:
        raise ValueError(f"{path} has {natoms} atoms, but topology has {topology_natoms} atoms.")

    start_offset = math.floor((1.0 - last_fraction) * len(selected_indices))
    retained_indices = set(selected_indices[start_offset:])
    last_retained_index = selected_indices[-1]
    start_frame = selected_indices[start_offset]

    center_groups, neighbor_groups = build_groups(
        molecules,
        center_resnames=center_resnames,
        neighbor_resnames=neighbor_resnames,
        target_indices=target_indices,
        center_mode=center_mode,
    )
    histograms = {
        (center_name, neighbor_name): np.zeros(edges_a.size - 1, dtype=float)
        for center_name in center_groups
        for neighbor_name in neighbor_groups
    }
    neighbor_ids_needed = np.unique(np.concatenate(list(neighbor_groups.values())))

    kept_frames = 0
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

            positions = np.empty((natoms, 3), dtype=float)
            for atom_idx in range(natoms):
                positions[atom_idx] = parse_position_line(handle.readline())

            center_positions = molecule_centers(
                positions,
                molecules,
                box_length=box_length,
                reference_mode="mol_com",
            )
            neighbor_positions = molecule_centers(
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
                    distances = minimum_image_pair_distances(
                        centers,
                        neighbors,
                        center_ids=center_ids,
                        neighbor_ids=neighbor_ids,
                        box_length=box_length,
                    )
                    histograms[(center_name, neighbor_name)] += np.histogram(distances, bins=edges_a)[0]
            kept_frames += 1
            frame_idx += 1

    if kept_frames == 0:
        raise ValueError(f"No frames retained from {path}.")

    shell_volumes = (4.0 / 3.0) * np.pi * (edges_a[1:] ** 3 - edges_a[:-1] ** 3)
    rdf: dict[tuple[str, str], np.ndarray] = {}
    for key, hist in histograms.items():
        center_name, neighbor_name = key
        center_ids = center_groups[center_name]
        neighbor_ids = neighbor_groups[neighbor_name]
        valid_pairs = center_ids.size * neighbor_ids.size - len(set(center_ids).intersection(set(neighbor_ids)))
        if valid_pairs <= 0:
            raise ValueError(f"No valid molecule pairs for {center_name}-{neighbor_name}.")
        normalization = kept_frames * valid_pairs * shell_volumes / box_length**3
        rdf[key] = hist / normalization

    return {
        "dataset": dataset_name,
        "path": str(path),
        "natoms": natoms,
        "n_frames_total": len(selected_indices),
        "start_frame": start_frame,
        "n_frames_used": kept_frames,
        "rdf": rdf,
        "center_groups": center_groups,
        "neighbor_groups": neighbor_groups,
        "neighbor_center": normalize_reference_mode(neighbor_center),
    }


def write_rdf_csv(
    aimd: dict[str, object],
    mlip: dict[str, object],
    r_mid_nm: np.ndarray,
    path: Path,
) -> None:
    center_groups = aimd["center_groups"]
    neighbor_groups = aimd["neighbor_groups"]
    fields = ["r_nm"]
    for center_name in center_groups:
        for neighbor_name in neighbor_groups:
            fields.append(f"AIMD_{center_name}_{neighbor_name}")
            fields.append(f"MLIP_{center_name}_{neighbor_name}")

    aimd_rdf = aimd["rdf"]
    mlip_rdf = mlip["rdf"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bin_idx, radius in enumerate(r_mid_nm):
            row: dict[str, float] = {"r_nm": float(radius)}
            for center_name in center_groups:
                for neighbor_name in neighbor_groups:
                    key = (center_name, neighbor_name)
                    row[f"AIMD_{center_name}_{neighbor_name}"] = float(aimd_rdf[key][bin_idx])
                    row[f"MLIP_{center_name}_{neighbor_name}"] = float(mlip_rdf[key][bin_idx])
            writer.writerow(row)


def plot_rdfs(
    aimd: dict[str, object],
    mlip: dict[str, object],
    r_mid_nm: np.ndarray,
    path: Path,
    *,
    r_max_nm: float,
) -> None:
    center_groups = list(aimd["center_groups"])
    neighbor_groups = list(aimd["neighbor_groups"])
    aimd_rdf = aimd["rdf"]
    mlip_rdf = mlip["rdf"]
    n_panels = len(center_groups) * len(neighbor_groups)

    def display_center(name: str) -> str:
        return name[4:] if name.startswith("all_") else name

    fig, axes = plt.subplots(
        len(center_groups),
        len(neighbor_groups),
        figsize=(3.8 * len(neighbor_groups), 3.2 * len(center_groups)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for row_idx, center_name in enumerate(center_groups):
        for col_idx, neighbor_name in enumerate(neighbor_groups):
            ax = axes[row_idx, col_idx]
            key = (center_name, neighbor_name)
            neighbor_label = reference_mode_label(neighbor_name, str(aimd["neighbor_center"]))
            ax.plot(r_mid_nm, aimd_rdf[key], color="#1f2937", linewidth=1.7, label=str(aimd["dataset"]))
            ax.plot(r_mid_nm, mlip_rdf[key], color="#2563eb", linewidth=1.7, label=str(mlip["dataset"]))
            ax.set_title(f"{display_center(center_name)}-{neighbor_label}")
            ax.set_xlim(0.0, r_max_nm)
            ax.grid(True, color="#d4d4d8", linewidth=0.7, alpha=0.7)
            if col_idx == 0:
                ax.set_ylabel("g(r)")
            if row_idx == len(center_groups) - 1:
                ax.set_xlabel("r (nm)")
            if n_panels == 1 or (row_idx == 0 and col_idx == len(neighbor_groups) - 1):
                ax.legend(frameon=False)
    if n_panels > 1:
        fig.suptitle(
            "Molecule COM RDF comparison "
            f"(AIMD {aimd['n_frames_used']} frames, MLIP {mlip['n_frames_used']} frames)"
        )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(
    aimd: dict[str, object],
    mlip: dict[str, object],
    molecules: list[Molecule],
    path: Path,
    args: argparse.Namespace,
    *,
    box_length: float,
) -> None:
    res_counts: dict[str, int] = {}
    for molecule in molecules:
        res_counts[molecule.resname] = res_counts.get(molecule.resname, 0) + 1
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"AIMD trajectory: {aimd['path']}\n")
        handle.write(f"MLIP trajectory: {mlip['path']}\n")
        handle.write(f"topology PDB: {args.top_pdb}\n")
        handle.write(f"cell: cubic {box_length:.6f} A\n")
        handle.write(f"rmax: {args.r_max_nm:.4f} nm\n")
        handle.write(f"dr: {args.dr_nm:.4f} nm\n")
        handle.write(f"last_fraction: {args.last_fraction:.3f}\n")
        handle.write(f"max_time_ps: {args.max_time_ps}\n")
        handle.write(f"target_indices: {args.target_indices}\n")
        handle.write(f"center_resnames: {args.center_resnames}\n")
        handle.write(f"neighbor_resnames: {args.neighbor_resnames}\n")
        handle.write(f"neighbor_center: {args.neighbor_center}\n")
        handle.write(f"molecules_by_resname: {res_counts}\n\n")
        for data in (aimd, mlip):
            handle.write(f"[{data['dataset']}]\n")
            handle.write(f"total frames considered: {data['n_frames_total']}\n")
            handle.write(f"start frame: {data['start_frame']}\n")
            handle.write(f"frames used: {data['n_frames_used']}\n")
            handle.write(f"center_groups: { {k: v.tolist() for k, v in data['center_groups'].items()} }\n")
            handle.write(f"neighbor_groups: { {k: v.tolist() for k, v in data['neighbor_groups'].items()} }\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AIMD and MLIP-MD molecule-COM RDFs over the last R fraction."
    )
    parser.add_argument("--aimd-xyz", type=Path, default=DEFAULT_AIMD_XYZ)
    parser.add_argument("--mlip-xyz", type=Path, required=True)
    parser.add_argument("--top-pdb", type=Path, default=DEFAULT_TOP_PDB)
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
    parser.add_argument("--last-fraction", type=float, default=0.20)
    parser.add_argument(
        "--max-time-ps",
        type=float,
        default=None,
        help="Keep only initial frames with parsed time <= this value before applying --last-fraction.",
    )
    parser.add_argument("--r-max-nm", type=float, default=0.75)
    parser.add_argument("--dr-nm", type=float, default=0.01)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--prefix", default="T5_mol_com_rdf_compare")
    args = parser.parse_args()
    if not args.aimd_xyz.is_file():
        raise FileNotFoundError(args.aimd_xyz)
    if not args.mlip_xyz.is_file():
        raise FileNotFoundError(args.mlip_xyz)
    if not args.top_pdb.is_file():
        raise FileNotFoundError(args.top_pdb)
    if not 0.0 < args.last_fraction <= 1.0:
        parser.error("--last-fraction must be in (0, 1].")
    if args.r_max_nm <= 0.0 or args.dr_nm <= 0.0:
        parser.error("--r-max-nm and --dr-nm must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive.")
    if args.max_time_ps is not None and args.max_time_ps <= 0.0:
        parser.error("--max-time-ps must be positive.")
    if args.cell_length_a is not None and args.cell_length_a <= 0.0:
        parser.error("--cell-length-a must be positive.")
    try:
        args.neighbor_center = normalize_reference_mode(args.neighbor_center)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    center_resnames = parse_csv_list(args.center_resnames)
    neighbor_resnames = parse_csv_list(args.neighbor_resnames)
    target_indices = parse_indices(args.target_indices)
    molecules, topology_natoms, topology_box_length = load_topology(args.top_pdb)
    box_length = args.cell_length_a if args.cell_length_a is not None else topology_box_length
    if box_length is None:
        raise ValueError("No cell length provided and no cubic CRYST1 record found in top PDB.")

    out_dir = args.out_dir or args.mlip_xyz.parent / "T5_mol_com_rdf_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_a = np.arange(0.0, args.r_max_nm * 10.0 + args.dr_nm * 10.0, args.dr_nm * 10.0)
    r_mid_nm = 0.5 * (edges_a[:-1] + edges_a[1:]) / 10.0

    aimd = accumulate_mol_com_rdfs(
        args.aimd_xyz,
        dataset_name=args.aimd_label,
        molecules=molecules,
        topology_natoms=topology_natoms,
        box_length=box_length,
        last_fraction=args.last_fraction,
        edges_a=edges_a,
        center_resnames=center_resnames,
        neighbor_resnames=neighbor_resnames,
        target_indices=target_indices,
        center_mode=args.center_mode,
        neighbor_center=args.neighbor_center,
        max_frames=args.max_frames,
        max_time_ps=args.max_time_ps,
    )
    mlip = accumulate_mol_com_rdfs(
        args.mlip_xyz,
        dataset_name=args.mlip_label,
        molecules=molecules,
        topology_natoms=topology_natoms,
        box_length=box_length,
        last_fraction=args.last_fraction,
        edges_a=edges_a,
        center_resnames=center_resnames,
        neighbor_resnames=neighbor_resnames,
        target_indices=target_indices,
        center_mode=args.center_mode,
        neighbor_center=args.neighbor_center,
        max_frames=args.max_frames,
        max_time_ps=args.max_time_ps,
    )

    plot_path = out_dir / f"{args.prefix}.png"
    csv_path = out_dir / f"{args.prefix}.csv"
    summary_path = out_dir / f"{args.prefix}.txt"
    plot_rdfs(aimd, mlip, r_mid_nm, plot_path, r_max_nm=args.r_max_nm)
    write_rdf_csv(aimd, mlip, r_mid_nm, csv_path)
    write_summary(aimd, mlip, molecules, summary_path, args, box_length=box_length)
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
