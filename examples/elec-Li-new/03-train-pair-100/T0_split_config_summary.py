from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import iread, read


DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100")
TEST_DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/electrolyte")
DEFAULT_PAIR_FILE = DATA_ROOT / "Li_system_lambda_parent_del_pairs.xyz"
DEFAULT_TEST_FILE = TEST_DATA_ROOT / "Li_system_test_with_del.xyz"
PAIR_ROLE_KEYS = ("delta_pair_role", "lambda_pair_role")


def normal_config_frame(atoms: Atoms) -> tuple[str, int]:
    for config_key in ("config_type", "lambda_selected_config_type", "lambda_parent_config_type"):
        if config_key in atoms.info:
            config = str(atoms.info[config_key])
            break
    else:
        raise KeyError("Normal structure is missing config metadata.")

    for frame_key in ("frame", "delta_pair_parent_frame", "lambda_source_parent_index"):
        if frame_key in atoms.info:
            frame = int(atoms.info[frame_key])
            break
    else:
        raise KeyError("Normal structure is missing frame/source-index metadata.")
    return config, frame


def deleted_parent_config_frame(atoms: Atoms) -> tuple[str, int]:
    for config_key in ("delta_pair_parent_config_type", "deleted_parent_config_type", "lambda_parent_config_type"):
        if config_key in atoms.info:
            config = str(atoms.info[config_key])
            break
    else:
        raise KeyError("Deleted structure is missing parent config metadata.")

    for frame_key in ("delta_pair_parent_frame", "deleted_parent_frame", "lambda_source_parent_index"):
        if frame_key in atoms.info:
            frame = int(atoms.info[frame_key])
            break
    else:
        raise KeyError("Deleted structure is missing parent frame metadata.")
    return config, frame


def pair_role(atoms: Atoms) -> int:
    for key in PAIR_ROLE_KEYS:
        if key in atoms.info:
            return int(atoms.info[key])
    raise KeyError(f"Missing pair role metadata; expected one of {PAIR_ROLE_KEYS}.")


def add_record(
    records: dict[tuple[str, str], list[int]],
    split: str,
    config_type: str,
    frame: int,
) -> None:
    records[(split, config_type)].append(int(frame))


def summarize_records(records: dict[tuple[str, str], list[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split, config_type), frames in sorted(records.items()):
        unique_frames = sorted(set(frames))
        rows.append(
            {
                "split": split,
                "config_type": config_type,
                "n_records": len(frames),
                "n_unique_frames": len(unique_frames),
                "frame_min": min(unique_frames),
                "frame_max": max(unique_frames),
                "frame_values": " ".join(str(frame) for frame in unique_frames),
            }
        )
    return rows


def load_pair_split_records(
    pair_file: Path,
    *,
    train_split: float,
    shuffle_seed: int,
    max_parent_frame: int | None,
) -> tuple[dict[tuple[str, str], list[int]], dict[str, Any]]:
    atoms_list = read(pair_file, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if len(atoms_list) % 2:
        raise ValueError(f"Pair file must contain an even number of structures: {pair_file}")

    original_n_pairs = len(atoms_list) // 2
    if max_parent_frame is not None:
        filtered_atoms: list[Atoms] = []
        for pair_index in range(original_n_pairs):
            normal = atoms_list[2 * pair_index]
            _config_type, frame = normal_config_frame(normal)
            if frame < max_parent_frame:
                filtered_atoms.extend([normal, atoms_list[2 * pair_index + 1]])
        atoms_list = filtered_atoms

    n_pairs = len(atoms_list) // 2
    pair_indices = np.arange(n_pairs)
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(pair_indices)
    train_len = int(train_split * n_pairs)
    split_by_pair = {
        int(pair_index): "train" if position < train_len else "val"
        for position, pair_index in enumerate(pair_indices)
    }

    records: dict[tuple[str, str], list[int]] = defaultdict(list)
    for pair_index in range(n_pairs):
        normal = atoms_list[2 * pair_index]
        deleted = atoms_list[2 * pair_index + 1]
        if pair_role(normal) != 0:
            raise ValueError(f"Expected normal role at pair {pair_index}")
        if pair_role(deleted) != 1:
            raise ValueError(f"Expected deleted role at pair {pair_index}")
        config_type, frame = normal_config_frame(normal)
        add_record(records, f"{split_by_pair[pair_index]}_pairs", config_type, frame)

    metadata = {
        "pair_file": str(pair_file),
        "n_pair_structures": len(atoms_list),
        "n_pairs": n_pairs,
        "n_pairs_before_parent_frame_filter": original_n_pairs,
        "max_parent_frame": max_parent_frame,
        "train_split": train_split,
        "shuffle_seed": shuffle_seed,
        "n_train_pairs": int(train_len),
        "n_val_pairs": int(n_pairs - train_len),
    }
    return records, metadata


def load_test_records(test_file: Path) -> tuple[dict[tuple[str, str], list[int]], dict[str, Any]]:
    records: dict[tuple[str, str], list[int]] = defaultdict(list)
    n_normal = 0
    n_deleted = 0
    for atoms in iread(test_file, ":"):
        if "config_type" in atoms.info:
            config_type, frame = normal_config_frame(atoms)
            add_record(records, "test_normal_structures", config_type, frame)
            n_normal += 1
        else:
            config_type, frame = deleted_parent_config_frame(atoms)
            add_record(records, "test_deleted_pairs", config_type, frame)
            n_deleted += 1

    metadata = {
        "test_file": str(test_file),
        "n_test_normal_structures": n_normal,
        "n_test_deleted_structures": n_deleted,
    }
    return records, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "config_type",
        "n_records",
        "n_unique_frames",
        "frame_min",
        "frame_max",
        "frame_values",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize train/val/test config_type and frame coverage."
    )
    parser.add_argument("--pair-file", type=Path, default=DEFAULT_PAIR_FILE)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--max-parent-frame",
        type=int,
        default=-1,
        help="Keep only train/val pairs whose normal parent frame is below this value. Set <0 to disable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "test_outputs",
    )
    parser.add_argument("--prefix", default="T0_split_config_summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_parent_frame = None if args.max_parent_frame < 0 else args.max_parent_frame
    pair_records, pair_metadata = load_pair_split_records(
        args.pair_file,
        train_split=args.train_split,
        shuffle_seed=args.shuffle_seed,
        max_parent_frame=max_parent_frame,
    )
    test_records, test_metadata = load_test_records(args.test_file)
    merged: dict[tuple[str, str], list[int]] = defaultdict(list)
    for source in (pair_records, test_records):
        for key, values in source.items():
            merged[key].extend(values)

    rows = summarize_records(merged)
    csv_path = args.out_dir / f"{args.prefix}.csv"
    json_path = args.out_dir / f"{args.prefix}.json"
    write_csv(csv_path, rows)
    summary = {
        **pair_metadata,
        **test_metadata,
        "rows": rows,
        "csv": str(csv_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(
        "pairs: "
        f"train={pair_metadata['n_train_pairs']} "
        f"val={pair_metadata['n_val_pairs']} "
        f"test_normal={test_metadata['n_test_normal_structures']} "
        f"test_deleted={test_metadata['n_test_deleted_structures']}"
    )


if __name__ == "__main__":
    main()
