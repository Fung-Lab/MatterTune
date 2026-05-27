from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from tqdm import tqdm

import mattertune.configs as MC
from mattertune.main import load_pretrained_model


DEFAULT_DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100")
DEFAULT_TRAIN_FILE = DEFAULT_DATA_ROOT / "Li_system_lambda_parent_del_pairs.xyz"
MODEL_TYPES = ("mattersim", "orb", "uma")


def normalize_model_type(raw: str) -> str:
    model_type = raw.strip().lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type {raw!r}; expected one of {MODEL_TYPES}.")
    return model_type


def normalize_model_name(model_type: str, model_name: str) -> str:
    model_type = normalize_model_type(model_type)
    name = model_name.strip()
    if model_type == "orb":
        aliases = {
            "orbv3-omat-conservative-inf": "orb-v3-conservative-inf-omat",
            "orb-v3-omat-conservative-inf": "orb-v3-conservative-inf-omat",
            "orbv3-conservative-inf-omat": "orb-v3-conservative-inf-omat",
        }
        return aliases.get(name, name.replace("_", "-"))
    if model_type == "uma":
        aliases = {
            "uma-s1.1": "uma-s-1p1",
            "uma-s-1.1": "uma-s-1p1",
            "uma-s1p1": "uma-s-1p1",
            "uma-s1.2": "uma-s-1p2",
            "uma-s-1.2": "uma-s-1p2",
            "uma-s1p2": "uma-s-1p2",
        }
        return aliases.get(name, name)
    return name


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
    model_type: str,
    model_name: str,
    task_name: str,
    device: str,
) -> np.ndarray:
    load_kwargs = {}
    if model_type == "uma":
        load_kwargs["task_name"] = task_name
    model = load_pretrained_model(
        model_type=model_type,
        model_name=model_name,
        device=device,
        **load_kwargs,
    )
    calc = model.ase_calculator()

    energies: list[float] = []
    for atoms in tqdm(atoms_list, desc=f"{model_type} pretrained energies"):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        energies.append(float(atoms_copy.get_potential_energy()))
    return np.asarray(energies, dtype=np.float64)


def _move_to_device(value, device: torch.device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_move_to_device(item, device) for item in value))
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def create_training_energy_model(
    *,
    model_type: str,
    model_name: str,
    task_name: str,
    graph_radius: float,
    max_num_neighbors: int,
    orb_edge_method: str | None,
):
    if model_type == "mattersim":
        model_config = MC.MatterSimBackboneConfig.draft()
        model_config.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        model_config.pretrained_model = model_name
    elif model_type == "orb":
        model_config = MC.ORBBackboneConfig.draft()
        model_config.pretrained_model = model_name
        model_config.system = MC.ORBSystemConfig(
            radius=graph_radius,
            max_num_neighbors=max_num_neighbors,
            edge_method=orb_edge_method,
        )
    elif model_type == "uma":
        model_config = MC.UMABackboneConfig.draft()
        model_config.model_name = model_name
        model_config.task_name = task_name
        model_config.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig(
            radius=graph_radius,
            max_num_neighbors=max_num_neighbors,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model_config.properties = [
        MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(),
            loss_coefficient=1.0,
        )
    ]
    if model_type == "uma":
        # UMA uses the energy+force head when conservative forces are part of
        # training. The raw energy baseline must come from that same head.
        model_config.properties.append(
            MC.ForcesPropertyConfig(
                loss=MC.MSELossConfig(),
                loss_coefficient=1.0,
                conservative=True,
            )
        )
    model_config.optimizer = MC.AdamWConfig(lr=1.0e-3)
    model_config.reset_output_heads = False
    model_config.freeze_backbone = False
    model_config.ignore_gpu_batch_transform_error = True
    model_config = model_config.finalize(strict=False)
    model_config.ensure_dependencies()
    model = model_config.create_model()
    model.eval()
    return model


def training_head_energies(
    atoms_list: list[Atoms],
    *,
    model_type: str,
    model_name: str,
    task_name: str,
    device: str,
    graph_radius: float,
    max_num_neighbors: int,
    orb_edge_method: str | None,
    batch_size: int,
) -> np.ndarray:
    torch_device = torch.device(device)
    model = create_training_energy_model(
        model_type=model_type,
        model_name=model_name,
        task_name=task_name,
        graph_radius=graph_radius,
        max_num_neighbors=max_num_neighbors,
        orb_edge_method=orb_edge_method,
    )
    model.to(torch_device)

    energies: list[float] = []
    batch_size = max(1, int(batch_size))
    iterator = range(0, len(atoms_list), batch_size)
    for start in tqdm(iterator, desc=f"{model_type} training-head raw energies"):
        chunk = atoms_list[start : start + batch_size]
        data_list = [
            model.cpu_data_transform(model.atoms_to_data(atoms, has_labels=False))
            for atoms in chunk
        ]
        batch = model.collate_fn(data_list)
        batch = _move_to_device(batch, torch_device)
        with torch.enable_grad():
            output = model(batch, mode="predict", ignore_gpu_batch_transform_error=False)
        energy = output["predicted_properties"]["energy"].detach().cpu().reshape(-1)
        energies.extend(float(value) for value in energy)

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
    parser.add_argument("--model_type", choices=MODEL_TYPES, default="mattersim")
    parser.add_argument("--model_name", default="MatterSim-v1.0.0-1M")
    parser.add_argument("--task_name", default="omat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reference_energy_source",
        choices=("ase_pretrained", "training_head"),
        default="ase_pretrained",
        help=(
            "Energy baseline to subtract before fitting composition references. "
            "Use training_head when the finetuning module's raw energy head is not "
            "the same quantity as the pretrained ASE calculator energy."
        ),
    )
    parser.add_argument("--graph_radius", type=float, default=6.0)
    parser.add_argument("--max_num_neighbors", type=int, default=120)
    parser.add_argument(
        "--orb_edge_method",
        choices=("knn_brute_force", "knn_scipy", "knn_cuml_brute", "knn_cuml_rbc", "knn_alchemi"),
        default=None,
        help=(
            "Optional ORB graph edge-construction method for training-head "
            "baseline evaluation. knn_scipy avoids nvalchemiops/Warp CUDA "
            "initialization noise during CPU featurization."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
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
    args.model_type = normalize_model_type(args.model_type)
    args.model_name = normalize_model_name(args.model_type, args.model_name)
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
    if args.reference_energy_source == "training_head":
        baseline_energies = training_head_energies(
            atoms_list,
            model_type=args.model_type,
            model_name=args.model_name,
            task_name=args.task_name,
            device=args.device,
            graph_radius=args.graph_radius,
            max_num_neighbors=args.max_num_neighbors,
            orb_edge_method=args.orb_edge_method,
            batch_size=args.batch_size,
        )
    else:
        baseline_energies = pretrained_energies(
            atoms_list,
            model_type=args.model_type,
            model_name=args.model_name,
            task_name=args.task_name,
            device=args.device,
        )
    residual = dft_energies - baseline_energies
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
            / f"{args.xyz_path.stem}-{args.model_type}-{args.model_name}-residual-energy_reference.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(references, handle, indent=4, sort_keys=True)

    summary = {
        "xyz_path": str(args.xyz_path),
        "frames": len(atoms_list),
        "model_type": args.model_type,
        "model_name": args.model_name,
        "task_name": args.task_name,
        "reference_energy_source": args.reference_energy_source,
        "graph_radius": args.graph_radius,
        "max_num_neighbors": args.max_num_neighbors,
        "orb_edge_method": args.orb_edge_method,
        "reference_model": args.reference_model,
        "ridge_alpha": args.ridge_alpha,
        "mean_dft_energy_eV": float(dft_energies.mean()),
        "mean_baseline_energy_eV": float(baseline_energies.mean()),
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
