from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write


DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte")
OUTPUT_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100")

SOLVENT_FORMULAS: dict[str, dict[str, int]] = {
    "G2": {"C": 6, "H": 14, "O": 3},
    "DME": {"C": 4, "H": 10, "O": 2},
    "FEC": {"C": 3, "F": 1, "H": 3, "O": 3},
    "EC": {"C": 3, "H": 4, "O": 3},
    "THF": {"C": 4, "H": 8, "O": 1},
    "PC": {"C": 4, "H": 6, "O": 3},
}
SOLVENT_CASES = {
    "G2": "case1",
    "DME": "case2",
    "FEC": "case3",
    "EC": "case4",
    "THF": "case5",
    "PC": "case6",
}


@dataclass(frozen=True)
class SelectedPair:
    row: dict[str, str]
    config_type: str
    selection_policy: str
    selection_rank: int
    inferred_config: bool


def read_all(path: Path) -> list[Atoms]:
    atoms = read(path, index=":")
    if not isinstance(atoms, list):
        return [atoms]
    return atoms


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def composition_signature(atoms: Atoms) -> str:
    counts = Counter(atoms.get_chemical_symbols())
    return " ".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def parse_config(config_type: str) -> tuple[str, str, float]:
    parts = config_type.split("-")
    if len(parts) < 7 or parts[2] != "Li" or parts[3] != "FSI":
        raise ValueError(f"Unsupported config_type format: {config_type}")
    concentration_case = parts[0]
    solvent = parts[4]
    ratio = float(parts[-1])
    return concentration_case, solvent, ratio


def infer_solvent_and_ratio(atoms: Atoms) -> tuple[str, float]:
    counts = Counter(atoms.get_chemical_symbols())
    n_fsi = counts["N"]
    if n_fsi <= 0:
        raise ValueError(f"Cannot infer LiFSI count from composition {dict(counts)}.")

    residual = Counter(counts)
    for symbol, multiplier in {"F": 2, "N": 1, "O": 4, "S": 2}.items():
        residual[symbol] -= multiplier * n_fsi
    residual["Li"] = 0
    residual = Counter({symbol: count for symbol, count in residual.items() if count})

    matches: dict[str, int] = {}
    for solvent, formula in SOLVENT_FORMULAS.items():
        candidate_counts: list[int] = []
        ok = True
        for symbol, formula_count in formula.items():
            count = residual.get(symbol, 0)
            if count % formula_count:
                ok = False
                break
            candidate_counts.append(count // formula_count)
        if not ok or not candidate_counts:
            continue
        n_solvent = candidate_counts[0]
        if any(count != n_solvent for count in candidate_counts):
            continue
        reconstructed = Counter(
            {symbol: formula_count * n_solvent for symbol, formula_count in formula.items()}
        )
        if all(
            residual.get(symbol, 0) == reconstructed.get(symbol, 0)
            for symbol in set(residual) | set(reconstructed)
        ):
            matches[solvent] = int(n_solvent)

    if len(matches) != 1:
        raise ValueError(f"Could not infer solvent from composition {dict(counts)}; matches={matches}")
    solvent, n_solvent = next(iter(matches.items()))
    return solvent, float(n_solvent) / float(n_fsi)


def build_config_maps(lambda0_atoms: list[Atoms]) -> tuple[dict[str, str], dict[str, list[float]], list[dict[str, Any]]]:
    composition_to_config: dict[str, str] = {}
    solvent_ratios: dict[str, set[float]] = defaultdict(set)
    rows: list[dict[str, Any]] = []

    for atoms in lambda0_atoms:
        config = str(atoms.info.get("config_type", ""))
        if not config:
            raise ValueError("lambda0 atom is missing config_type.")
        signature = composition_signature(atoms)
        existing = composition_to_config.get(signature)
        if existing is not None and existing != config:
            raise ValueError(
                f"Composition maps to multiple config types: {signature}: {existing}, {config}"
            )
        composition_to_config[signature] = config
        _concentration_case, solvent, ratio = parse_config(config)
        solvent_ratios[solvent].add(ratio)

    for signature, config in sorted(composition_to_config.items(), key=lambda item: item[1]):
        _concentration_case, solvent, ratio = parse_config(config)
        rows.append(
            {
                "composition": signature,
                "config_type": config,
                "solvent": solvent,
                "solvent_ratio": ratio,
                "source": "lambda0_metadata",
            }
        )

    return composition_to_config, {key: sorted(values, reverse=True) for key, values in solvent_ratios.items()}, rows


def infer_config_type(
    atoms: Atoms,
    *,
    composition_to_config: dict[str, str],
    solvent_ratios: dict[str, list[float]],
) -> tuple[str, bool]:
    signature = composition_signature(atoms)
    if signature in composition_to_config:
        return composition_to_config[signature], False

    solvent, ratio = infer_solvent_and_ratio(atoms)
    observed = list(solvent_ratios.get(solvent, []))
    merged = sorted(set(observed + [ratio]), reverse=True)
    concentration_case = f"case{merged.index(ratio) + 1}"
    solvent_case = SOLVENT_CASES[solvent]
    return f"{concentration_case}-{solvent_case}-Li-FSI-{solvent}-1-{ratio:.1f}", True


def load_source_atoms(data_root: Path) -> dict[str, list[Atoms]]:
    return {
        "lambda0_parent": read_all(data_root / "Li_system_lambda0.xyz"),
        "lambda0_deleted": read_all(data_root / "Li_system_lambda0_del.xyz"),
        "lambda1_parent": read_all(data_root / "Li_system_lambda1.xyz"),
        "lambda1_deleted": read_all(data_root / "Li_system_lambda1_del.xyz"),
    }


def select_lambda0_pairs(rows: list[dict[str, str]], *, per_config_count: int) -> list[SelectedPair]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["lambda_label"] != "lambda0":
            continue
        config = row["parent_config_type"]
        if not config:
            raise ValueError(f"lambda0 row is missing parent_config_type: {row}")
        grouped[config].append(row)

    selected: list[SelectedPair] = []
    for config in sorted(grouped):
        config_rows = sorted(
            grouped[config],
            key=lambda row: (int(row["parent_frame"]), int(row["parent_index"])),
        )
        if len(config_rows) < per_config_count:
            raise ValueError(
                f"lambda0 config {config} has only {len(config_rows)} matched pairs; "
                f"need {per_config_count}."
            )
        for rank, row in enumerate(config_rows[:per_config_count]):
            selected.append(
                SelectedPair(
                    row=row,
                    config_type=config,
                    selection_policy="lambda0_first_matched_parent_frames",
                    selection_rank=rank,
                    inferred_config=False,
                )
            )
    return selected


def select_lambda1_pairs(
    rows: list[dict[str, str]],
    *,
    source_atoms: dict[str, list[Atoms]],
    composition_to_config: dict[str, str],
    solvent_ratios: dict[str, list[float]],
    per_config_count: int,
    seed: int,
) -> tuple[list[SelectedPair], list[dict[str, Any]]]:
    grouped: dict[str, list[tuple[dict[str, str], bool]]] = defaultdict(list)
    inferred_rows: dict[str, dict[str, Any]] = {}
    lambda1_parent = source_atoms["lambda1_parent"]

    for row in rows:
        if row["lambda_label"] != "lambda1":
            continue
        parent = lambda1_parent[int(row["parent_index"])]
        config, inferred = infer_config_type(
            parent,
            composition_to_config=composition_to_config,
            solvent_ratios=solvent_ratios,
        )
        grouped[config].append((row, inferred))
        if inferred:
            signature = composition_signature(parent)
            inferred_rows[signature] = {
                "composition": signature,
                "config_type": config,
                "solvent": infer_solvent_and_ratio(parent)[0],
                "solvent_ratio": infer_solvent_and_ratio(parent)[1],
                "source": "inferred_from_composition",
            }

    rng = np.random.default_rng(seed)
    selected: list[SelectedPair] = []
    for config in sorted(grouped):
        config_rows = grouped[config]
        if len(config_rows) < per_config_count:
            raise ValueError(
                f"lambda1 config {config} has only {len(config_rows)} matched pairs; "
                f"need {per_config_count}."
            )
        chosen_indices = rng.choice(len(config_rows), size=per_config_count, replace=False)
        chosen = [config_rows[int(index)] for index in chosen_indices]
        chosen.sort(key=lambda item: int(item[0]["parent_index"]))
        for rank, (row, inferred) in enumerate(chosen):
            selected.append(
                SelectedPair(
                    row=row,
                    config_type=config,
                    selection_policy=f"lambda1_random_{per_config_count}_per_config_seed_{seed}",
                    selection_rank=rank,
                    inferred_config=inferred,
                )
            )

    return selected, sorted(inferred_rows.values(), key=lambda item: item["config_type"])


def annotate_parent(atoms: Atoms, pair: SelectedPair) -> Atoms:
    copied = copy_with_results(atoms)
    row = pair.row
    copied.info["lambda_label"] = row["lambda_label"]
    copied.info["lambda_pair_id"] = int(row["pair_id"])
    copied.info["lambda_pair_role"] = 0
    copied.info["lambda_source_parent_index"] = int(row["parent_index"])
    copied.info["lambda_source_deleted_index"] = int(row["deleted_index"])
    copied.info["lambda_selected_config_type"] = pair.config_type
    copied.info["lambda_selection_policy"] = pair.selection_policy
    copied.info["lambda_selection_rank"] = pair.selection_rank
    if "config_type" not in copied.info:
        copied.info["config_type"] = pair.config_type
    return copied


def annotate_deleted(atoms: Atoms, pair: SelectedPair) -> Atoms:
    copied = copy_with_results(atoms)
    row = pair.row
    copied.info["lambda_label"] = row["lambda_label"]
    copied.info["lambda_pair_id"] = int(row["pair_id"])
    copied.info["lambda_pair_role"] = 1
    copied.info["lambda_source_parent_index"] = int(row["parent_index"])
    copied.info["lambda_source_deleted_index"] = int(row["deleted_index"])
    copied.info["lambda_parent_config_type"] = pair.config_type
    copied.info["lambda_selection_policy"] = pair.selection_policy
    copied.info["lambda_selection_rank"] = pair.selection_rank
    copied.info["deleted_parent_config_type"] = pair.config_type
    return copied


def materialize_pairs(
    selected: list[SelectedPair],
    *,
    source_atoms: dict[str, list[Atoms]],
) -> tuple[dict[str, list[Atoms]], list[Atoms], list[dict[str, Any]]]:
    output_atoms = {
        "lambda0_parent": [],
        "lambda0_deleted": [],
        "lambda1_parent": [],
        "lambda1_deleted": [],
    }
    paired_atoms: list[Atoms] = []
    mapping_rows: list[dict[str, Any]] = []

    for output_index, pair in enumerate(selected):
        row = pair.row
        label = row["lambda_label"]
        parent_key = f"{label}_parent"
        deleted_key = f"{label}_deleted"
        parent = annotate_parent(source_atoms[parent_key][int(row["parent_index"])], pair)
        deleted = annotate_deleted(source_atoms[deleted_key][int(row["deleted_index"])], pair)

        output_atoms[parent_key].append(parent)
        output_atoms[deleted_key].append(deleted)
        paired_atoms.extend([parent, deleted])

        mapping_row: dict[str, Any] = dict(row)
        mapping_row.update(
            {
                "subset_pair_index": output_index,
                "selection_config_type": pair.config_type,
                "selection_policy": pair.selection_policy,
                "selection_rank_within_config": pair.selection_rank,
                "config_inferred_from_composition": pair.inferred_config,
            }
        )
        mapping_rows.append(mapping_row)

    return output_atoms, paired_atoms, mapping_rows


def summarize_selected(selected: list[SelectedPair]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[SelectedPair]] = defaultdict(list)
    for pair in selected:
        grouped[(pair.row["lambda_label"], pair.config_type)].append(pair)

    rows: list[dict[str, Any]] = []
    for (label, config), pairs in sorted(grouped.items()):
        parent_frames = [
            int(pair.row["parent_frame"])
            for pair in pairs
            if pair.row.get("parent_frame", "") != ""
        ]
        parent_indices = [int(pair.row["parent_index"]) for pair in pairs]
        rows.append(
            {
                "lambda_label": label,
                "config_type": config,
                "n_pairs": len(pairs),
                "n_inferred_config": sum(pair.inferred_config for pair in pairs),
                "parent_frame_min": min(parent_frames) if parent_frames else None,
                "parent_frame_max": max(parent_frames) if parent_frames else None,
                "parent_index_min": min(parent_indices),
                "parent_index_max": max(parent_indices),
            }
        )
    return rows


def build_subset(args: argparse.Namespace) -> dict[str, Any]:
    source_atoms = load_source_atoms(args.data_root)
    pair_rows = read_csv(args.pair_mapping)
    composition_to_config, solvent_ratios, config_map_rows = build_config_maps(
        source_atoms["lambda0_parent"]
    )

    selected_lambda0 = select_lambda0_pairs(pair_rows, per_config_count=args.per_config_count)
    selected_lambda1, inferred_config_rows = select_lambda1_pairs(
        pair_rows,
        source_atoms=source_atoms,
        composition_to_config=composition_to_config,
        solvent_ratios=solvent_ratios,
        per_config_count=args.per_config_count,
        seed=args.seed,
    )
    selected = selected_lambda0 + selected_lambda1
    output_atoms, paired_atoms, selected_mapping_rows = materialize_pairs(
        selected,
        source_atoms=source_atoms,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "lambda0_parent": args.output_root / "Li_system_lambda0.xyz",
        "lambda0_deleted": args.output_root / "Li_system_lambda0_del.xyz",
        "lambda1_parent": args.output_root / "Li_system_lambda1.xyz",
        "lambda1_deleted": args.output_root / "Li_system_lambda1_del.xyz",
        "paired": args.output_root / "Li_system_lambda_parent_del_pairs.xyz",
        "mapping": args.output_root / "Li_system_lambda_parent_del_pair_mapping.csv",
        "summary": args.output_root / "Li_system_lambda100_subset_summary.json",
        "config_map": args.output_root / "Li_system_lambda_config_formula_mapping.csv",
    }

    for key in ("lambda0_parent", "lambda0_deleted", "lambda1_parent", "lambda1_deleted"):
        write(output_paths[key], output_atoms[key], format="extxyz")
    write(output_paths["paired"], paired_atoms, format="extxyz")
    write_csv(output_paths["mapping"], selected_mapping_rows)
    write_csv(output_paths["config_map"], config_map_rows + inferred_config_rows)

    by_config = summarize_selected(selected)
    summary = {
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "pair_mapping": str(args.pair_mapping),
        "per_config_count": args.per_config_count,
        "lambda1_random_seed": args.seed,
        "outputs": {key: str(path) for key, path in output_paths.items()},
        "totals": {
            "n_lambda0_pairs": len(selected_lambda0),
            "n_lambda1_pairs": len(selected_lambda1),
            "n_pairs": len(selected),
            "n_paired_structures": len(paired_atoms),
            "n_config_types_lambda0": len({pair.config_type for pair in selected_lambda0}),
            "n_config_types_lambda1": len({pair.config_type for pair in selected_lambda1}),
            "n_lambda1_pairs_with_inferred_config": sum(pair.inferred_config for pair in selected_lambda1),
        },
        "by_config": by_config,
    }
    output_paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 100-pair-per-config subset from lambda0/lambda1 parent-deleted pairs. "
            "lambda0 uses earliest matched parent frames; lambda1 samples random pairs per config."
        )
    )
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--pair_mapping",
        type=Path,
        default=DATA_ROOT / "Li_system_lambda_parent_del_pair_mapping.csv",
    )
    parser.add_argument("--per_config_count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_subset(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
