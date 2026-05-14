from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms


SOLVENT_FORMULAS: dict[str, dict[str, int]] = {
    "G2": {"C": 6, "H": 14, "O": 3},
    "DME": {"C": 4, "H": 10, "O": 2},
    "FEC": {"C": 3, "F": 1, "H": 3, "O": 3},
    "EC": {"C": 3, "H": 4, "O": 3},
    "THF": {"C": 4, "H": 8, "O": 1},
    "PC": {"C": 4, "H": 6, "O": 3},
}

FEATURE_NAMES = [
    "LiFSI",
    "G2",
    "DME",
    "FEC",
    "EC",
    "THF",
    "PC",
    "deleted_Li",
]


def infer_solvent_counts(atoms: Atoms) -> dict[str, int]:
    counts = Counter(atoms.get_chemical_symbols())
    n_fsi = counts["N"]
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
        reconstructed = Counter({symbol: formula_count * n_solvent for symbol, formula_count in formula.items()})
        if all(residual.get(symbol, 0) == reconstructed.get(symbol, 0) for symbol in set(residual) | set(reconstructed)):
            matches[solvent] = int(n_solvent)

    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely infer solvent for composition {dict(counts)}; matches={matches}"
        )
    return matches


def component_feature_vector(atoms: Atoms) -> np.ndarray:
    counts = Counter(atoms.get_chemical_symbols())
    n_fsi = int(counts["N"])
    n_li = int(counts["Li"])
    solvent_counts = infer_solvent_counts(atoms)
    deleted_li = int(round(n_fsi + 1 - n_li))
    if deleted_li not in (0, 1):
        raise ValueError(
            f"Expected deleted_Li feature to be 0 or 1, got {deleted_li} "
            f"for Li={n_li}, FSI={n_fsi}."
        )

    values = {
        "LiFSI": n_fsi,
        "deleted_Li": deleted_li,
        **solvent_counts,
    }
    return np.asarray([values.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)


def component_matrix(atoms_list: list[Atoms]) -> np.ndarray:
    return np.stack([component_feature_vector(atoms) for atoms in atoms_list])


def load_component_reference(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    feature_names = list(payload["feature_names"])
    coefficients = np.asarray([payload["coefficients"][name] for name in feature_names], dtype=np.float64)
    return feature_names, coefficients


def component_reference_energy(atoms: Atoms, feature_names: list[str], coefficients: np.ndarray) -> float:
    features = component_feature_vector(atoms)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    ordered_features = np.asarray([features[feature_index[name]] for name in feature_names], dtype=np.float64)
    return float(ordered_features @ coefficients)


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(value) for value in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
