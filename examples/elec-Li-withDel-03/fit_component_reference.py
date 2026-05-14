from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm

from mattertune.main import load_pretrained_model

from component_reference import FEATURE_NAMES, component_matrix, json_sanitize


DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/electrolyte")
DEFAULT_TRAIN_FILE = DATA_ROOT / "Li_system_train_with_del.xyz"


def read_all(path: Path) -> list[Atoms]:
    atoms_list = read(path, index=":")
    if not isinstance(atoms_list, list):
        return [atoms_list]
    return atoms_list


def pretrained_energies(atoms_list: list[Atoms], *, model_name: str, device: str) -> np.ndarray:
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


def fit_coefficients(
    features: np.ndarray,
    residual: np.ndarray,
    *,
    reference_model: str,
    ridge_alpha: float,
) -> np.ndarray:
    if reference_model == "linear":
        model = LinearRegression(fit_intercept=False)
    elif reference_model == "ridge":
        model = Ridge(fit_intercept=False, alpha=ridge_alpha)
    else:
        raise ValueError(f"Unsupported reference model: {reference_model}")
    return np.asarray(model.fit(features, residual).coef_, dtype=np.float64)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    atoms_list = read_all(args.xyz_path)
    if args.limit is not None:
        atoms_list = atoms_list[: args.limit]
    if not atoms_list:
        raise ValueError(f"No structures loaded from {args.xyz_path}")

    dft = np.asarray([float(atoms.get_potential_energy()) for atoms in atoms_list], dtype=np.float64)
    pretrained = pretrained_energies(atoms_list, model_name=args.model_name, device=args.device)
    residual = dft - pretrained
    features = component_matrix(atoms_list)
    coefficients = fit_coefficients(
        features,
        residual,
        reference_model=args.reference_model,
        ridge_alpha=args.ridge_alpha,
    )
    fitted = features @ coefficients
    residual_after_ref = residual - fitted

    output = args.output
    if output is None:
        output = (
            Path(__file__).resolve().parent
            / "data"
            / f"{args.xyz_path.stem}-{args.model_name}-component-residual-{args.reference_model}-alpha{args.ridge_alpha}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "feature_names": FEATURE_NAMES,
        "coefficients": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, coefficients, strict=True)
        },
        "reference_model": args.reference_model,
        "ridge_alpha": args.ridge_alpha,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, sort_keys=True)

    summary = {
        "xyz_path": args.xyz_path,
        "frames": len(atoms_list),
        "model_name": args.model_name,
        "reference_model": args.reference_model,
        "ridge_alpha": args.ridge_alpha,
        "feature_names": FEATURE_NAMES,
        "feature_rank": int(np.linalg.matrix_rank(features)),
        "mean_dft_energy_eV": float(np.mean(dft)),
        "mean_pretrained_energy_eV": float(np.mean(pretrained)),
        "mean_residual_energy_eV": float(np.mean(residual)),
        "residual_after_reference_mae_eV": float(np.mean(np.abs(residual_after_ref))),
        "residual_after_reference_rmse_eV": float(np.sqrt(np.mean(residual_after_ref**2))),
        "coefficients": payload["coefficients"],
    }
    summary_path = output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_sanitize(summary), handle, indent=4, sort_keys=True)

    logging.info("Saved component residual reference to %s", output)
    logging.info("Saved summary to %s", summary_path)
    logging.info("Frames: %d", len(atoms_list))
    logging.info("Residual-after-reference MAE: %.12f eV", summary["residual_after_reference_mae_eV"])
    logging.info("Residual-after-reference RMSE: %.12f eV", summary["residual_after_reference_rmse_eV"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz_path", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model_name", default="MatterSim-v1.0.0-1M")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference_model", choices=("linear", "ridge"), default="ridge")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.xyz_path.is_file():
        raise FileNotFoundError(args.xyz_path)
    return args


if __name__ == "__main__":
    main(parse_args())
