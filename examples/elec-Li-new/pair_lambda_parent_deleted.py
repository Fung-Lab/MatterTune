from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from scipy.optimize import linear_sum_assignment


DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte")
DEFAULT_OUTPUT_PREFIX = "Li_system_lambda_parent_del"


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    parent_file: Path
    deleted_file: Path


@dataclass(frozen=True)
class ParentRecord:
    label: str
    index: int
    atoms: Atoms


@dataclass(frozen=True)
class DeletedRecord:
    label: str
    index: int
    atoms: Atoms


def read_all(path: Path) -> list[Atoms]:
    atoms = read(path, index=":")
    if not isinstance(atoms, list):
        return [atoms]
    return atoms


def species_key(atoms: Atoms, *, remove_index: int | None = None) -> tuple[str, ...]:
    return tuple(
        atom.symbol
        for index, atom in enumerate(atoms)
        if remove_index is None or index != remove_index
    )


def species_key_digest(key: tuple[str, ...]) -> str:
    return hashlib.sha1("|".join(key).encode("utf-8")).hexdigest()[:12]


def lattice_lengths(atoms: Atoms) -> np.ndarray:
    cell = np.asarray(atoms.cell.array, dtype=np.float32)
    if cell.shape == (3, 3) and np.allclose(cell, np.diag(np.diag(cell)), atol=1.0e-8):
        return np.diag(cell).astype(np.float32)
    return np.asarray([1.0e9, 1.0e9, 1.0e9], dtype=np.float32)


def rmsd_matrix(a: np.ndarray, b: np.ndarray, lengths: np.ndarray, block: int = 32) -> np.ndarray:
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    for start in range(0, a.shape[0], block):
        delta = a[start : start + block, None, :, :] - b[None, :, :, :]
        delta -= lengths[None, :, None, :] * np.round(delta / lengths[None, :, None, :])
        out[start : start + block] = np.sqrt(
            np.mean(np.sum(delta * delta, axis=3), axis=2, dtype=np.float32)
        )
    return out


def atoms_energy(atoms: Atoms) -> float:
    return float(atoms.get_potential_energy())


def atoms_forces(atoms: Atoms) -> np.ndarray:
    try:
        return np.asarray(atoms.get_forces(), dtype=np.float64).copy()
    except PropertyNotImplementedError:
        if "forces" in atoms.arrays:
            return np.asarray(atoms.arrays["forces"], dtype=np.float64).copy()
        if "force" in atoms.arrays:
            return np.asarray(atoms.arrays["force"], dtype=np.float64).copy()
        raise


def copy_with_results(atoms: Atoms) -> Atoms:
    copied = atoms.copy()
    copied.calc = SinglePointCalculator(
        copied,
        energy=atoms_energy(atoms),
        forces=atoms_forces(atoms),
    )
    return copied


def stable_pair_id(label: str, parent_index: int, deleted_index: int) -> int:
    digest = hashlib.sha1(f"{label}|{parent_index}|{deleted_index}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def build_parent_groups(
    *,
    label: str,
    parent_atoms: list[Atoms],
) -> dict[tuple[str, ...], list[ParentRecord]]:
    parent_groups: dict[tuple[str, ...], list[ParentRecord]] = defaultdict(list)
    for index, atoms in enumerate(parent_atoms):
        symbols = atoms.get_chemical_symbols()
        if not symbols or symbols[0] != "Li":
            raise ValueError(f"{label} parent frame {index} does not have Li at index 0.")
        parent_groups[species_key(atoms, remove_index=0)].append(
            ParentRecord(label=label, index=index, atoms=atoms)
        )
    return parent_groups


def build_deleted_groups(
    *,
    label: str,
    deleted_atoms: list[Atoms],
    parent_groups: dict[tuple[str, ...], list[ParentRecord]],
    deleted_file: Path,
) -> tuple[dict[tuple[str, ...], list[DeletedRecord]], list[dict[str, Any]]]:
    deleted_groups: dict[tuple[str, ...], list[DeletedRecord]] = defaultdict(list)
    unmatched_deleted_rows: list[dict[str, Any]] = []
    for index, atoms in enumerate(deleted_atoms):
        key = species_key(atoms)
        if key not in parent_groups:
            unmatched_deleted_rows.append(
                unmatched_deleted_row(
                    label=label,
                    deleted_file=deleted_file,
                    deleted_index=index,
                    atoms=atoms,
                    reason="no_parent_species_key",
                    species_key=key,
                )
            )
            continue
        deleted_groups[key].append(DeletedRecord(label=label, index=index, atoms=atoms))
    return deleted_groups, unmatched_deleted_rows


def metadata_value(atoms: Atoms, key: str) -> str:
    value = atoms.info.get(key, "")
    return "" if value is None else str(value)


def mapping_row(
    *,
    label: str,
    parent_file: Path,
    deleted_file: Path,
    parent: ParentRecord,
    deleted: DeletedRecord,
    distance: float,
    pair_id: int,
) -> dict[str, Any]:
    return {
        "lambda_label": label,
        "pair_id": pair_id,
        "parent_file": str(parent_file),
        "parent_index": parent.index,
        "deleted_file": str(deleted_file),
        "deleted_index": deleted.index,
        "parent_atom_count": len(parent.atoms),
        "deleted_atom_count": len(deleted.atoms),
        "parent_config_type": metadata_value(parent.atoms, "config_type"),
        "parent_frame": metadata_value(parent.atoms, "frame"),
        "parent_energy_eV": atoms_energy(parent.atoms),
        "deleted_energy_eV": atoms_energy(deleted.atoms),
        "match_rmsd_A": distance,
        "species_key_sha1_12": species_key_digest(species_key(deleted.atoms)),
    }


def unmatched_parent_row(
    *,
    label: str,
    parent_file: Path,
    parent: ParentRecord,
    reason: str,
) -> dict[str, Any]:
    key = species_key(parent.atoms, remove_index=0)
    return {
        "lambda_label": label,
        "parent_file": str(parent_file),
        "parent_index": parent.index,
        "parent_atom_count": len(parent.atoms),
        "parent_config_type": metadata_value(parent.atoms, "config_type"),
        "parent_frame": metadata_value(parent.atoms, "frame"),
        "parent_energy_eV": atoms_energy(parent.atoms),
        "reason": reason,
        "species_key_sha1_12": species_key_digest(key),
    }


def unmatched_deleted_row(
    *,
    label: str,
    deleted_file: Path,
    deleted_index: int,
    atoms: Atoms,
    reason: str,
    species_key: tuple[str, ...],
    best_rmsd_A: float | None = None,
) -> dict[str, Any]:
    return {
        "lambda_label": label,
        "deleted_file": str(deleted_file),
        "deleted_index": deleted_index,
        "deleted_atom_count": len(atoms),
        "deleted_energy_eV": atoms_energy(atoms),
        "reason": reason,
        "best_rmsd_A": "" if best_rmsd_A is None else best_rmsd_A,
        "species_key_sha1_12": species_key_digest(species_key),
    }


def add_pair_metadata(
    *,
    parent: ParentRecord,
    deleted: DeletedRecord,
    label: str,
    parent_file: Path,
    deleted_file: Path,
    pair_id: int,
    distance: float,
) -> tuple[Atoms, Atoms]:
    parent_copy = copy_with_results(parent.atoms)
    deleted_copy = copy_with_results(deleted.atoms)

    common = {
        "lambda_label": label,
        "lambda_pair_id": pair_id,
        "lambda_parent_file": str(parent_file),
        "lambda_parent_index": parent.index,
        "lambda_deleted_file": str(deleted_file),
        "lambda_deleted_index": deleted.index,
        "lambda_match_rmsd_A": distance,
    }
    parent_copy.info.update(common)
    parent_copy.info["lambda_pair_role"] = 0

    deleted_copy.info.update(common)
    deleted_copy.info["lambda_pair_role"] = 1
    deleted_copy.info["lambda_parent_config_type"] = metadata_value(parent.atoms, "config_type")
    deleted_copy.info["lambda_parent_frame"] = metadata_value(parent.atoms, "frame")
    return parent_copy, deleted_copy


def match_one_dataset(
    spec: DatasetSpec,
    *,
    tolerance: float,
) -> tuple[list[Atoms], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    parent_atoms = read_all(spec.parent_file)
    deleted_atoms = read_all(spec.deleted_file)
    parent_groups = build_parent_groups(label=spec.label, parent_atoms=parent_atoms)
    deleted_groups, unmatched_deleted_rows = build_deleted_groups(
        label=spec.label,
        deleted_atoms=deleted_atoms,
        parent_groups=parent_groups,
        deleted_file=spec.deleted_file,
    )

    paired_atoms: list[Atoms] = []
    mapping_rows: list[dict[str, Any]] = []
    unmatched_parent_rows: list[dict[str, Any]] = []
    matched_parent_ids: set[int] = set()
    matched_deleted_ids: set[int] = set()
    distances: list[float] = []

    for key, parent_records in sorted(parent_groups.items()):
        deleted_records = deleted_groups.get(key, [])
        if not deleted_records:
            unmatched_parent_rows.extend(
                unmatched_parent_row(
                    label=spec.label,
                    parent_file=spec.parent_file,
                    parent=parent,
                    reason="no_deleted_species_key",
                )
                for parent in parent_records
            )
            continue

        deleted_positions = np.stack(
            [record.atoms.positions.astype(np.float32) for record in deleted_records]
        )
        parent_positions = np.stack(
            [record.atoms.positions[1:].astype(np.float32) for record in parent_records]
        )
        parent_lengths = np.stack([lattice_lengths(record.atoms) for record in parent_records])
        distance_matrix = rmsd_matrix(deleted_positions, parent_positions, parent_lengths)
        rows, cols = linear_sum_assignment(distance_matrix)

        assigned_parent_local: set[int] = set()
        assigned_deleted_local: set[int] = set()
        best_deleted_distance = distance_matrix.min(axis=1)
        best_parent_distance = distance_matrix.min(axis=0)

        for row, col in zip(rows, cols, strict=True):
            distance = float(distance_matrix[row, col])
            if distance > tolerance:
                continue

            deleted = deleted_records[row]
            parent = parent_records[col]
            pair_id = stable_pair_id(spec.label, parent.index, deleted.index)
            parent_copy, deleted_copy = add_pair_metadata(
                parent=parent,
                deleted=deleted,
                label=spec.label,
                parent_file=spec.parent_file,
                deleted_file=spec.deleted_file,
                pair_id=pair_id,
                distance=distance,
            )
            paired_atoms.extend([parent_copy, deleted_copy])
            mapping_rows.append(
                mapping_row(
                    label=spec.label,
                    parent_file=spec.parent_file,
                    deleted_file=spec.deleted_file,
                    parent=parent,
                    deleted=deleted,
                    distance=distance,
                    pair_id=pair_id,
                )
            )
            assigned_deleted_local.add(row)
            assigned_parent_local.add(col)
            matched_deleted_ids.add(deleted.index)
            matched_parent_ids.add(parent.index)
            distances.append(distance)

        for row, deleted in enumerate(deleted_records):
            if row in assigned_deleted_local:
                continue
            unmatched_deleted_rows.append(
                unmatched_deleted_row(
                    label=spec.label,
                    deleted_file=spec.deleted_file,
                    deleted_index=deleted.index,
                    atoms=deleted.atoms,
                    reason="no_parent_within_tolerance",
                    species_key=key,
                    best_rmsd_A=float(best_deleted_distance[row]),
                )
            )
        for col, parent in enumerate(parent_records):
            if col in assigned_parent_local:
                continue
            unmatched_parent_rows.append(
                unmatched_parent_row(
                    label=spec.label,
                    parent_file=spec.parent_file,
                    parent=parent,
                    reason="no_deleted_within_tolerance",
                )
            )

    summary = {
        "lambda_label": spec.label,
        "parent_file": str(spec.parent_file),
        "deleted_file": str(spec.deleted_file),
        "n_parent_frames": len(parent_atoms),
        "n_deleted_file_frames": len(deleted_atoms),
        "n_pairs": len(mapping_rows),
        "n_paired_structures": len(paired_atoms),
        "n_unmatched_parents": len(unmatched_parent_rows),
        "n_unmatched_deleted": len(unmatched_deleted_rows),
        "n_deleted_no_parent_species_key": sum(
            row["reason"] == "no_parent_species_key" for row in unmatched_deleted_rows
        ),
        "max_match_rmsd_A": max(distances) if distances else None,
        "mean_match_rmsd_A": float(np.mean(distances)) if distances else None,
        "match_tolerance_A": tolerance,
    }
    return paired_atoms, mapping_rows, unmatched_parent_rows, unmatched_deleted_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_specs(data_root: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            label="lambda0",
            parent_file=data_root / "Li_system_lambda0.xyz",
            deleted_file=data_root / "Li_system_lambda0_del.xyz",
        ),
        DatasetSpec(
            label="lambda1",
            parent_file=data_root / "Li_system_lambda1.xyz",
            deleted_file=data_root / "Li_system_lambda1_del.xyz",
        ),
    ]


def pair_all(args: argparse.Namespace) -> dict[str, Any]:
    specs = default_specs(args.data_root)
    for spec in specs:
        for path in (spec.parent_file, spec.deleted_file):
            if not path.is_file():
                raise FileNotFoundError(path)

    all_paired_atoms: list[Atoms] = []
    all_mapping_rows: list[dict[str, Any]] = []
    all_unmatched_parent_rows: list[dict[str, Any]] = []
    all_unmatched_deleted_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for spec in specs:
        paired_atoms, mapping_rows, unmatched_parent_rows, unmatched_deleted_rows, summary = (
            match_one_dataset(spec, tolerance=args.match_tolerance)
        )
        all_paired_atoms.extend(paired_atoms)
        all_mapping_rows.extend(mapping_rows)
        all_unmatched_parent_rows.extend(unmatched_parent_rows)
        all_unmatched_deleted_rows.extend(unmatched_deleted_rows)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_file = args.output_dir / f"{args.output_prefix}_pairs.xyz"
    mapping_file = args.output_dir / f"{args.output_prefix}_pair_mapping.csv"
    unmatched_parent_file = args.output_dir / f"{args.output_prefix}_unmatched_parents.csv"
    unmatched_deleted_file = args.output_dir / f"{args.output_prefix}_unmatched_deleted.csv"
    summary_file = args.output_dir / f"{args.output_prefix}_pair_summary.json"

    if not all_paired_atoms:
        raise ValueError("No parent/deleted pairs were matched.")

    write(pair_file, all_paired_atoms, format="extxyz")
    write_csv(mapping_file, all_mapping_rows)
    write_csv(unmatched_parent_file, all_unmatched_parent_rows)
    write_csv(unmatched_deleted_file, all_unmatched_deleted_rows)

    summary = {
        "output_pairs": str(pair_file),
        "output_mapping": str(mapping_file),
        "output_unmatched_parents": str(unmatched_parent_file),
        "output_unmatched_deleted": str(unmatched_deleted_file),
        "match_tolerance_A": args.match_tolerance,
        "datasets": summaries,
        "totals": {
            "n_pairs": len(all_mapping_rows),
            "n_paired_structures": len(all_paired_atoms),
            "n_unmatched_parents": len(all_unmatched_parent_rows),
            "n_unmatched_deleted": len(all_unmatched_deleted_rows),
        },
    }
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair lambda0/lambda1 full parent structures with deleted-Li structures "
            "generated by removing atom index 0."
        )
    )
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--match_tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(pair_all(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
