from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, read, write
from scipy.optimize import linear_sum_assignment


DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/electrolyte")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_MAPPING = (
    REPO_ROOT
    / "examples"
    / "electrolyte"
    / "notes"
    / "li_delete_pair_mapping.csv"
)


def read_all(path: Path) -> list[Atoms]:
    atoms_list = read(path, index=":")
    if not isinstance(atoms_list, list):
        return [atoms_list]
    return atoms_list


def species_key(atoms: Atoms) -> tuple[str, ...]:
    return tuple(atoms.get_chemical_symbols())


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


def map_mixed_deleted_to_train_delete(
    mixed_deleted: list[tuple[int, Atoms]],
    train_delete: list[Atoms],
    *,
    tolerance: float,
) -> dict[int, int]:
    mixed_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    delete_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)

    for local_index, (_, atoms) in enumerate(mixed_deleted):
        mixed_groups[species_key(atoms)].append(local_index)
    for index, atoms in enumerate(train_delete):
        delete_groups[species_key(atoms)].append(index)

    mapping: dict[int, int] = {}
    for key, mixed_local_indices in mixed_groups.items():
        train_delete_indices = delete_groups.get(key)
        if not train_delete_indices:
            raise ValueError(f"No train_delete structures have species key of length {len(key)}.")

        mixed_positions = np.stack(
            [mixed_deleted[index][1].positions.astype(np.float32) for index in mixed_local_indices]
        )
        delete_positions = np.stack(
            [train_delete[index].positions.astype(np.float32) for index in train_delete_indices]
        )
        lengths = np.stack([lattice_lengths(train_delete[index]) for index in train_delete_indices])
        distances = rmsd_matrix(mixed_positions, delete_positions, lengths)

        rows, cols = linear_sum_assignment(distances)
        for row, col in zip(rows, cols, strict=True):
            distance = float(distances[row, col])
            if distance > tolerance:
                mixed_frame = mixed_deleted[mixed_local_indices[row]][0]
                raise ValueError(
                    f"Could not match mixed deleted frame {mixed_frame}: nearest RMSD "
                    f"{distance:.6g} A exceeds tolerance {tolerance}."
                )
            mapping[mixed_deleted[mixed_local_indices[row]][0]] = train_delete_indices[col]

    return mapping


def load_parent_mapping(path: Path) -> dict[int, dict[str, str]]:
    mapping: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[int(row["delete_frame_index"])] = row
    return mapping


def stable_pair_id(*parts: Any) -> int:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def copy_with_single_point_results(atoms: Atoms) -> Atoms:
    copied = atoms.copy()
    copied.calc = SinglePointCalculator(
        copied,
        energy=float(atoms.get_potential_energy()),
        forces=np.asarray(atoms.get_forces(), dtype=np.float64).copy(),
    )
    return copied


def prepare_delta_pairs(args: argparse.Namespace) -> None:
    mixed_atoms = read_all(args.mixed_train_file)
    train_delete_atoms = read_all(args.train_delete_file)
    parent_mapping = load_parent_mapping(args.pair_mapping_file)
    parent_sources = {piece.strip() for piece in args.parent_sources.split(",") if piece.strip()}

    parent_atoms_by_source = {
        "train": read_all(args.base_train_file),
        "test": read_all(args.base_test_file),
    }

    mixed_deleted = [
        (index, atoms)
        for index, atoms in enumerate(mixed_atoms)
        if "config_type" not in atoms.info
    ]
    mixed_to_train_delete = map_mixed_deleted_to_train_delete(
        mixed_deleted,
        train_delete_atoms,
        tolerance=args.match_tolerance,
    )

    paired: list[Atoms] = []
    skipped_by_source = 0
    for mixed_index, deleted_atoms in mixed_deleted:
        train_delete_index = mixed_to_train_delete[mixed_index]
        parent = parent_mapping[train_delete_index]
        parent_source = parent["original_file"]
        if parent_source not in parent_sources:
            skipped_by_source += 1
            continue

        parent_index = int(parent["original_frame_index_in_file"])
        parent_atoms = copy_with_single_point_results(parent_atoms_by_source[parent_source][parent_index])
        deleted_copy = copy_with_single_point_results(deleted_atoms)
        pair_id = stable_pair_id(parent_source, parent_index, mixed_index, train_delete_index)

        parent_atoms.info["delta_pair_id"] = pair_id
        parent_atoms.info["delta_pair_role"] = 0
        parent_atoms.info["delta_pair_deleted_mixed_index"] = mixed_index
        parent_atoms.info["delta_pair_train_delete_index"] = train_delete_index
        parent_atoms.info["delta_pair_parent_source"] = parent_source
        parent_atoms.info["delta_pair_parent_index"] = parent_index

        deleted_copy.info["delta_pair_id"] = pair_id
        deleted_copy.info["delta_pair_role"] = 1
        deleted_copy.info["delta_pair_deleted_mixed_index"] = mixed_index
        deleted_copy.info["delta_pair_train_delete_index"] = train_delete_index
        deleted_copy.info["delta_pair_parent_source"] = parent_source
        deleted_copy.info["delta_pair_parent_index"] = parent_index
        deleted_copy.info["delta_pair_parent_config_type"] = parent["original_config_type"]
        deleted_copy.info["delta_pair_parent_frame"] = parent["original_comment_frame"]

        paired.extend([parent_atoms, deleted_copy])

    if not paired:
        raise ValueError("No delta-E pairs were written. Check --parent_sources and input mappings.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, paired, format="extxyz")
    print(f"Wrote {len(paired) // 2} delta-E pairs ({len(paired)} structures) to {args.output}")
    print(f"Skipped {skipped_by_source} deleted structures by parent source filter: {args.parent_sources}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed_train_file", type=Path, default=DATA_ROOT / "Li_system_train_with_del.xyz")
    parser.add_argument("--base_train_file", type=Path, default=DATA_ROOT / "Li_system_train.xyz")
    parser.add_argument("--base_test_file", type=Path, default=DATA_ROOT / "Li_system_test.xyz")
    parser.add_argument("--train_delete_file", type=Path, default=DATA_ROOT / "train_delete.xyz")
    parser.add_argument("--pair_mapping_file", type=Path, default=DEFAULT_PAIR_MAPPING)
    parser.add_argument(
        "--parent_sources",
        default="train",
        help="Comma-separated parent sources to include. Default avoids adding normal test parents.",
    )
    parser.add_argument("--match_tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "local_runs" / "elec-Li-withDel-02" / "data" / "delta_pairs_train.xyz",
    )
    args = parser.parse_args()
    for path in (
        args.mixed_train_file,
        args.base_train_file,
        args.base_test_file,
        args.train_delete_file,
        args.pair_mapping_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    return args


if __name__ == "__main__":
    prepare_delta_pairs(parse_args())
