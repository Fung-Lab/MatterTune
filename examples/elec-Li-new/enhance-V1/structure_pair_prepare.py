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


DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1")
DEFAULT_OUTPUT_PREFIX = "Li_electrolyte_V1"


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


def metadata_value(atoms: Atoms, key: str) -> str:
    value = atoms.info.get(key, "")
    return "" if value is None else str(value)


def stable_pair_id(
    *,
    label: str,
    parent_file: Path,
    deleted_file: Path,
    parent_index: int,
    deleted_index: int,
) -> int:
    payload = f"{label}|{parent_file.name}|{deleted_file.name}|{parent_index}|{deleted_index}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def add_source_metadata(
    atoms: Atoms,
    *,
    label: str,
    source_file: Path,
    source_index: int,
    role: int,
) -> Atoms:
    copied = copy_with_results(atoms)
    copied.info.update(
        {
            "pair_label": label,
            "pair_role": role,
            "delta_pair_role": role,
            "lambda_pair_role": role,
            "pair_role_name": "without_del" if role == 0 else "with_del",
            "pair_source_file": str(source_file),
            "pair_source_index": source_index,
        }
    )
    return copied


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
    parent_copy = add_source_metadata(
        parent.atoms,
        label=label,
        source_file=parent_file,
        source_index=parent.index,
        role=0,
    )
    deleted_copy = add_source_metadata(
        deleted.atoms,
        label=label,
        source_file=deleted_file,
        source_index=deleted.index,
        role=1,
    )

    common = {
        "pair_id": pair_id,
        "delta_pair_id": pair_id,
        "lambda_pair_id": pair_id,
        "pair_parent_file": str(parent_file),
        "pair_parent_index": parent.index,
        "pair_deleted_file": str(deleted_file),
        "pair_deleted_index": deleted.index,
        "pair_match_rmsd_A": distance,
        "lambda_label": label,
        "lambda_parent_file": str(parent_file),
        "lambda_parent_index": parent.index,
        "lambda_deleted_file": str(deleted_file),
        "lambda_deleted_index": deleted.index,
        "lambda_match_rmsd_A": distance,
    }
    parent_copy.info.update(common)
    deleted_copy.info.update(common)
    deleted_copy.info["pair_parent_config_type"] = metadata_value(parent.atoms, "config_type")
    deleted_copy.info["pair_parent_frame"] = metadata_value(parent.atoms, "frame")
    deleted_copy.info["lambda_parent_config_type"] = metadata_value(parent.atoms, "config_type")
    deleted_copy.info["lambda_parent_frame"] = metadata_value(parent.atoms, "frame")
    return parent_copy, deleted_copy


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
        "pair_label": label,
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
        "delta_e_deleted_minus_parent_eV": atoms_energy(deleted.atoms) - atoms_energy(parent.atoms),
        "match_rmsd_A": distance,
        "species_key_sha1_12": species_key_digest(species_key(deleted.atoms)),
    }


def unmatched_parent_row(
    *,
    label: str,
    parent_file: Path,
    parent: ParentRecord,
    reason: str,
    removed_atom_index: int,
) -> dict[str, Any]:
    key = species_key(parent.atoms, remove_index=removed_atom_index)
    return {
        "pair_label": label,
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
    species_key_value: tuple[str, ...],
    best_rmsd_A: float | None = None,
) -> dict[str, Any]:
    return {
        "pair_label": label,
        "deleted_file": str(deleted_file),
        "deleted_index": deleted_index,
        "deleted_atom_count": len(atoms),
        "deleted_energy_eV": atoms_energy(atoms),
        "reason": reason,
        "best_rmsd_A": "" if best_rmsd_A is None else best_rmsd_A,
        "species_key_sha1_12": species_key_digest(species_key_value),
    }


def build_parent_groups(
    *,
    label: str,
    parent_atoms: list[Atoms],
    removed_atom_index: int,
    removed_symbol: str,
) -> dict[tuple[str, ...], list[ParentRecord]]:
    parent_groups: dict[tuple[str, ...], list[ParentRecord]] = defaultdict(list)
    for index, atoms in enumerate(parent_atoms):
        if removed_atom_index < 0 or removed_atom_index >= len(atoms):
            raise ValueError(
                f"{label} parent frame {index} does not contain removed atom index "
                f"{removed_atom_index}."
            )
        if removed_symbol and atoms[removed_atom_index].symbol != removed_symbol:
            raise ValueError(
                f"{label} parent frame {index} has {atoms[removed_atom_index].symbol} "
                f"at index {removed_atom_index}, expected {removed_symbol}."
            )
        parent_groups[species_key(atoms, remove_index=removed_atom_index)].append(
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
                    species_key_value=key,
                )
            )
            continue
        deleted_groups[key].append(DeletedRecord(label=label, index=index, atoms=atoms))
    return deleted_groups, unmatched_deleted_rows


def merged_atoms_for_spec(
    *,
    spec: DatasetSpec,
    parent_atoms: list[Atoms],
    deleted_atoms: list[Atoms],
) -> list[Atoms]:
    merged_atoms: list[Atoms] = []
    for index, atoms in enumerate(parent_atoms):
        merged_atoms.append(
            add_source_metadata(
                atoms,
                label=spec.label,
                source_file=spec.parent_file,
                source_index=index,
                role=0,
            )
        )
    for index, atoms in enumerate(deleted_atoms):
        merged_atoms.append(
            add_source_metadata(
                atoms,
                label=spec.label,
                source_file=spec.deleted_file,
                source_index=index,
                role=1,
            )
        )
    return merged_atoms


def match_one_dataset(
    spec: DatasetSpec,
    *,
    tolerance: float,
    removed_atom_index: int,
    removed_symbol: str,
) -> tuple[
    list[Atoms],
    list[Atoms],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    parent_atoms = read_all(spec.parent_file)
    deleted_atoms = read_all(spec.deleted_file)
    merged_atoms = merged_atoms_for_spec(
        spec=spec,
        parent_atoms=parent_atoms,
        deleted_atoms=deleted_atoms,
    )
    parent_groups = build_parent_groups(
        label=spec.label,
        parent_atoms=parent_atoms,
        removed_atom_index=removed_atom_index,
        removed_symbol=removed_symbol,
    )
    deleted_groups, unmatched_deleted_rows = build_deleted_groups(
        label=spec.label,
        deleted_atoms=deleted_atoms,
        parent_groups=parent_groups,
        deleted_file=spec.deleted_file,
    )

    paired_atoms: list[Atoms] = []
    mapping_rows: list[dict[str, Any]] = []
    unmatched_parent_rows: list[dict[str, Any]] = []
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
                    removed_atom_index=removed_atom_index,
                )
                for parent in parent_records
            )
            continue

        deleted_positions = np.stack(
            [record.atoms.positions.astype(np.float32) for record in deleted_records]
        )
        parent_positions = np.stack(
            [
                np.delete(record.atoms.positions, removed_atom_index, axis=0).astype(np.float32)
                for record in parent_records
            ]
        )
        parent_lengths = np.stack([lattice_lengths(record.atoms) for record in parent_records])
        distance_matrix = rmsd_matrix(deleted_positions, parent_positions, parent_lengths)
        rows, cols = linear_sum_assignment(distance_matrix)

        assigned_parent_local: set[int] = set()
        assigned_deleted_local: set[int] = set()
        best_deleted_distance = distance_matrix.min(axis=1)

        for row, col in zip(rows, cols, strict=True):
            distance = float(distance_matrix[row, col])
            if distance > tolerance:
                continue

            deleted = deleted_records[row]
            parent = parent_records[col]
            pair_id = stable_pair_id(
                label=spec.label,
                parent_file=spec.parent_file,
                deleted_file=spec.deleted_file,
                parent_index=parent.index,
                deleted_index=deleted.index,
            )
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
                    species_key_value=key,
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
                    removed_atom_index=removed_atom_index,
                )
            )

    summary = {
        "pair_label": spec.label,
        "parent_file": str(spec.parent_file),
        "deleted_file": str(spec.deleted_file),
        "n_parent_frames": len(parent_atoms),
        "n_deleted_file_frames": len(deleted_atoms),
        "n_pairs": len(mapping_rows),
        "n_paired_structures": len(paired_atoms),
        "n_merged_structures": len(merged_atoms),
        "n_unmatched_parents": len(unmatched_parent_rows),
        "n_unmatched_deleted": len(unmatched_deleted_rows),
        "n_deleted_no_parent_species_key": sum(
            row["reason"] == "no_parent_species_key" for row in unmatched_deleted_rows
        ),
        "max_match_rmsd_A": max(distances) if distances else None,
        "mean_match_rmsd_A": float(np.mean(distances)) if distances else None,
        "match_tolerance_A": tolerance,
    }
    return (
        paired_atoms,
        merged_atoms,
        mapping_rows,
        unmatched_parent_rows,
        unmatched_deleted_rows,
        summary,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("note\n", encoding="utf-8")
        return
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


def parent_stem_for_deleted(stem: str, deleted_suffix: str) -> str | None:
    if not stem.endswith(deleted_suffix):
        return None
    return stem[: -len(deleted_suffix)]


def discover_specs(
    data_root: Path,
    *,
    deleted_suffix: str,
    allow_missing_parent: bool,
) -> tuple[list[DatasetSpec], list[dict[str, str]]]:
    specs: list[DatasetSpec] = []
    missing_parent_rows: list[dict[str, str]] = []
    for deleted_file in sorted(data_root.glob(f"*{deleted_suffix}.xyz")):
        parent_stem = parent_stem_for_deleted(deleted_file.stem, deleted_suffix)
        if parent_stem is None:
            continue
        parent_file = deleted_file.with_name(f"{parent_stem}.xyz")
        if not parent_file.is_file():
            row = {
                "deleted_file": str(deleted_file),
                "expected_parent_file": str(parent_file),
                "reason": "missing_parent_file",
            }
            missing_parent_rows.append(row)
            if not allow_missing_parent:
                raise FileNotFoundError(
                    f"Missing parent file for {deleted_file}: expected {parent_file}"
                )
            continue
        specs.append(
            DatasetSpec(
                label=parent_stem,
                parent_file=parent_file,
                deleted_file=deleted_file,
            )
        )
    return specs, missing_parent_rows


def parse_label_filter(raw: str) -> set[str] | None:
    labels = {piece.strip() for piece in raw.replace(",", " ").split() if piece.strip()}
    return labels or None


def output_prefix(args: argparse.Namespace) -> str:
    if args.output_prefix:
        return args.output_prefix
    return args.data_root.name.replace("-", "_")


def pair_all(args: argparse.Namespace) -> dict[str, Any]:
    specs, missing_parent_rows = discover_specs(
        args.data_root,
        deleted_suffix=args.deleted_suffix,
        allow_missing_parent=args.allow_missing_parent,
    )
    include_labels = parse_label_filter(args.include_labels)
    if include_labels is not None:
        discovered_labels = {spec.label for spec in specs}
        missing_labels = sorted(include_labels - discovered_labels)
        if missing_labels:
            available = ", ".join(sorted(discovered_labels)) or "<none>"
            missing = ", ".join(missing_labels)
            raise ValueError(
                f"--include_labels requested labels not found: {missing}. "
                f"Available labels: {available}."
            )
        specs = [spec for spec in specs if spec.label in include_labels]
    if not specs:
        raise ValueError(
            f"No parent/deleted xyz pairs found in {args.data_root} with suffix "
            f"{args.deleted_suffix!r}."
        )

    all_paired_atoms: list[Atoms] = []
    all_merged_atoms: list[Atoms] = []
    all_mapping_rows: list[dict[str, Any]] = []
    all_unmatched_parent_rows: list[dict[str, Any]] = []
    all_unmatched_deleted_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for spec in specs:
        (
            paired_atoms,
            merged_atoms,
            mapping_rows,
            unmatched_parent_rows,
            unmatched_deleted_rows,
            summary,
        ) = match_one_dataset(
            spec,
            tolerance=args.match_tolerance,
            removed_atom_index=args.removed_atom_index,
            removed_symbol=args.removed_symbol,
        )
        all_paired_atoms.extend(paired_atoms)
        all_merged_atoms.extend(merged_atoms)
        all_mapping_rows.extend(mapping_rows)
        all_unmatched_parent_rows.extend(unmatched_parent_rows)
        all_unmatched_deleted_rows.extend(unmatched_deleted_rows)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix(args)
    pair_file = args.output_dir / f"{prefix}_pairs.xyz"
    merged_file = args.output_dir / f"{prefix}_all.xyz"
    mapping_file = args.output_dir / f"{prefix}_pair_mapping.csv"
    unmatched_parent_file = args.output_dir / f"{prefix}_unmatched_parents.csv"
    unmatched_deleted_file = args.output_dir / f"{prefix}_unmatched_deleted.csv"
    missing_parent_file = args.output_dir / f"{prefix}_missing_parent_files.csv"
    summary_file = args.output_dir / f"{prefix}_pair_summary.json"

    if not all_paired_atoms:
        raise ValueError("No parent/deleted pairs were matched.")
    if args.strict and (all_unmatched_parent_rows or all_unmatched_deleted_rows):
        raise ValueError(
            "Strict pairing requested, but unmatched structures exist. "
            f"Parents: {len(all_unmatched_parent_rows)}, deleted: {len(all_unmatched_deleted_rows)}."
        )

    write(pair_file, all_paired_atoms, format="extxyz")
    write(merged_file, all_merged_atoms, format="extxyz")
    write_csv(mapping_file, all_mapping_rows)
    write_csv(unmatched_parent_file, all_unmatched_parent_rows)
    write_csv(unmatched_deleted_file, all_unmatched_deleted_rows)
    write_csv(missing_parent_file, missing_parent_rows)

    summary = {
        "output_pairs": str(pair_file),
        "output_all": str(merged_file),
        "output_mapping": str(mapping_file),
        "output_unmatched_parents": str(unmatched_parent_file),
        "output_unmatched_deleted": str(unmatched_deleted_file),
        "output_missing_parent_files": str(missing_parent_file),
        "data_root": str(args.data_root),
        "deleted_suffix": args.deleted_suffix,
        "include_labels": None if include_labels is None else sorted(include_labels),
        "removed_atom_index": args.removed_atom_index,
        "removed_symbol": args.removed_symbol,
        "match_tolerance_A": args.match_tolerance,
        "datasets": summaries,
        "totals": {
            "n_input_file_pairs": len(specs),
            "n_pairs": len(all_mapping_rows),
            "n_paired_structures": len(all_paired_atoms),
            "n_merged_structures": len(all_merged_atoms),
            "n_unmatched_parents": len(all_unmatched_parent_rows),
            "n_unmatched_deleted": len(all_unmatched_deleted_rows),
            "n_missing_parent_files": len(missing_parent_rows),
        },
    }
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically pair XXX.xyz full structures with XXX_del.xyz "
            "deleted-Li structures for enhance-V1 delta-E training."
        )
    )
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--deleted_suffix",
        default="_del",
        help="Filename stem suffix used by deleted structures, e.g. XXX_del.xyz.",
    )
    parser.add_argument(
        "--include_labels",
        default="",
        help=(
            "Optional comma- or space-separated parent stems to include, "
            "e.g. Li_system_lambda0,Li_system_lambda1."
        ),
    )
    parser.add_argument("--removed_atom_index", type=int, default=0)
    parser.add_argument(
        "--removed_symbol",
        default="Li",
        help="Expected symbol at --removed_atom_index in parent structures. Set '' to disable.",
    )
    parser.add_argument("--match_tolerance", type=float, default=1.0e-4)
    parser.add_argument("--allow_missing_parent", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any parent or deleted structures remain unmatched.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.data_root
    return args


def main() -> None:
    print(json.dumps(pair_all(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
