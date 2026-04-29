from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.io import read
from tqdm import tqdm

from mattertune.main import load_pretrained_model


def composition_matrix(atoms_list: list[Atoms]) -> np.ndarray:
    num_elements = max(
        max(Counter(atoms.get_atomic_numbers()).keys()) for atoms in atoms_list
    ) + 1
    matrix = np.zeros((len(atoms_list), num_elements), dtype=np.float64)
    for row, atoms in enumerate(atoms_list):
        for atomic_number, count in Counter(atoms.get_atomic_numbers()).items():
            matrix[row, atomic_number] = count
    return matrix


def fit_references(
    compositions: np.ndarray,
    target_values: np.ndarray,
    *,
    reference_model: Literal["linear", "ridge"],
    alpha: float,
) -> dict[int, float]:
    if reference_model == "linear":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression(fit_intercept=False)
    elif reference_model == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(fit_intercept=False, alpha=alpha)
    else:
        raise ValueError(f"Unsupported reference model: {reference_model}")

    references = model.fit(compositions, target_values).coef_
    references_dict = {int(z): float(ref) for z, ref in enumerate(references.tolist())}
    references_dict.pop(0, None)
    return references_dict


def pretrained_energy(
    atoms_list: list[Atoms],
    *,
    model_type: str,
    model_name: str | None,
    device: str,
    model_kwargs: dict[str, Any],
) -> np.ndarray:
    model = load_pretrained_model(
        model_type,
        model_name,
        device=device,
        **model_kwargs,
    )
    calc = model.ase_calculator()

    energies: list[float] = []
    for atoms in tqdm(atoms_list, desc="Pretrained energies"):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        energies.append(float(atoms_copy.get_potential_energy()))
    return np.asarray(energies, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-atom residual references by fitting "
            "E_AIMD - E_pretrained against composition."
        )
    )
    parser.add_argument(
        "--xyz-path",
        default="/net/csefiles/coc-fung-cluster/lingyu/electrolyte/train.xyz",
        help="Training extxyz file used to fit residual references.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output JSON path. Defaults to ./data/"
            "<xyz-stem>-<model-name>-residual-energy_reference.json"
        ),
    )
    parser.add_argument(
        "--model-type",
        default="mattersim",
        help="Pretrained model family, for example mattersim or mace.",
    )
    parser.add_argument(
        "--model-name",
        default="MatterSim-v1.0.0-1M",
        help=(
            "Pretrained model name. Use mace-medium for the current MACE branch "
            "configuration."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reference-model",
        choices=("linear", "ridge"),
        default="ridge",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--model-kwargs",
        type=json.loads,
        default={},
        help="JSON dict of extra keyword arguments passed to load_pretrained_model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atoms_list: list[Atoms] = read(args.xyz_path, index=":")  # type: ignore[assignment]

    aimd_energies = np.asarray(
        [float(atoms.get_potential_energy()) for atoms in atoms_list],
        dtype=np.float64,
    )
    pretrained_energies = pretrained_energy(
        atoms_list,
        model_type=args.model_type,
        model_name=args.model_name,
        device=args.device,
        model_kwargs=args.model_kwargs,
    )
    residual_energies = aimd_energies - pretrained_energies
    compositions = composition_matrix(atoms_list)

    references = fit_references(
        compositions,
        residual_energies,
        reference_model=args.reference_model,
        alpha=args.ridge_alpha,
    )

    fitted_residual = compositions @ np.asarray(
        [references.get(z, 0.0) for z in range(compositions.shape[1])],
        dtype=np.float64,
    )
    normalized_target = aimd_energies - fitted_residual
    residual_error = normalized_target - pretrained_energies

    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parent
        / "data"
        / f"{Path(args.xyz_path).stem}-{args.model_name}-residual-energy_reference.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(references, handle, indent=4)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Saved residual energy reference to %s", output)
    logging.info("Frames: %d", len(atoms_list))
    logging.info("Mean AIMD energy: %.12f eV", float(aimd_energies.mean()))
    logging.info(
        "Mean pretrained energy: %.12f eV", float(pretrained_energies.mean())
    )
    logging.info(
        "Mean residual target: %.12f eV", float(residual_energies.mean())
    )
    logging.info(
        "After reference subtraction, target-pretrained MAE: %.12f eV",
        float(np.abs(residual_error).mean()),
    )
    logging.info(
        "After reference subtraction, target-pretrained RMSE: %.12f eV",
        float(np.sqrt(np.mean(residual_error**2))),
    )


if __name__ == "__main__":
    main()
