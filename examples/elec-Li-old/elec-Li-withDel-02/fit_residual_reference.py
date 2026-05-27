from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from tqdm import tqdm

from mattertune.main import load_pretrained_model


DEFAULT_DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/electrolyte")
DEFAULT_TRAIN_FILE = DEFAULT_DATA_ROOT / "Li_system_train_with_del.xyz"


def composition_matrix(atoms_list: list[Atoms]) -> np.ndarray:
    max_z = max(max(Counter(atoms.numbers).keys()) for atoms in atoms_list)
    matrix = np.zeros((len(atoms_list), max_z + 1), dtype=np.float64)
    for row, atoms in enumerate(atoms_list):
        for z, count in Counter(atoms.numbers).items():
            matrix[row, z] = count
    return matrix


def fit_references(
    compositions: np.ndarray,
    residual_energies: np.ndarray,
    *,
    reference_model: str,
    ridge_alpha: float,
) -> dict[int, float]:
    if reference_model == "linear":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression(fit_intercept=False)
    elif reference_model == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(fit_intercept=False, alpha=ridge_alpha)
    else:
        raise ValueError(f"Unsupported reference model: {reference_model}")

    coeffs = model.fit(compositions, residual_energies).coef_
    references = {int(z): float(ref) for z, ref in enumerate(coeffs.tolist())}
    references.pop(0, None)
    return references


def pretrained_energies(
    atoms_list: list[Atoms],
    *,
    model_name: str,
    device: str,
) -> np.ndarray:
    model = load_pretrained_model(
        model_type="mattersim",
        model_name=model_name,
        device=device,
    )
    calc = model.ase_calculator()

    energies: list[float] = []
    for atoms in tqdm(atoms_list, desc="MatterSim pretrained energies"):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        energies.append(float(atoms_copy.get_potential_energy()))
    return np.asarray(energies, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-element references to E_DFT - E_pretrained for the "
            "Li mixed normal+deleted dataset."
        )
    )
    parser.add_argument("--xyz_path", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model_name", default="MatterSim-v1.0.0-1M")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference_model", choices=("linear", "ridge"), default="ridge")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional quick-debug limit on the number of structures used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    atoms_list: list[Atoms] = read(args.xyz_path, index=":")  # type: ignore[assignment]
    if args.limit is not None:
        atoms_list = atoms_list[: args.limit]
    if not atoms_list:
        raise ValueError(f"No structures loaded from {args.xyz_path}")

    dft_energies = np.asarray(
        [float(atoms.get_potential_energy()) for atoms in atoms_list],
        dtype=np.float64,
    )
    pt_energies = pretrained_energies(
        atoms_list,
        model_name=args.model_name,
        device=args.device,
    )
    residual = dft_energies - pt_energies
    compositions = composition_matrix(atoms_list)

    references = fit_references(
        compositions,
        residual,
        reference_model=args.reference_model,
        ridge_alpha=args.ridge_alpha,
    )

    ref_vector = np.asarray(
        [references.get(z, 0.0) for z in range(compositions.shape[1])],
        dtype=np.float64,
    )
    fitted_residual = compositions @ ref_vector
    residual_after_ref = residual - fitted_residual

    output = args.output
    if output is None:
        output = (
            Path(__file__).resolve().parent
            / "data"
            / f"{args.xyz_path.stem}-{args.model_name}-residual-energy_reference.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(references, handle, indent=4, sort_keys=True)

    summary = {
        "xyz_path": str(args.xyz_path),
        "frames": len(atoms_list),
        "model_name": args.model_name,
        "reference_model": args.reference_model,
        "ridge_alpha": args.ridge_alpha,
        "mean_dft_energy_eV": float(dft_energies.mean()),
        "mean_pretrained_energy_eV": float(pt_energies.mean()),
        "mean_residual_energy_eV": float(residual.mean()),
        "residual_after_reference_mae_eV": float(np.abs(residual_after_ref).mean()),
        "residual_after_reference_rmse_eV": float(
            np.sqrt(np.mean(residual_after_ref**2))
        ),
        "references": references,
    }
    summary_path = output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4, sort_keys=True)

    logging.info("Saved residual energy reference to %s", output)
    logging.info("Saved summary to %s", summary_path)
    logging.info("Frames: %d", len(atoms_list))
    logging.info("Residual-after-reference MAE: %.12f eV", summary["residual_after_reference_mae_eV"])
    logging.info("Residual-after-reference RMSE: %.12f eV", summary["residual_after_reference_rmse_eV"])


if __name__ == "__main__":
    main()
