from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import read


@dataclass
class TimingRecord:
    prepare_ms: float
    backbone_ms: float
    autograd_ms: float
    total_ms: float


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clone_atomic_data(data: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone().detach()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def filter_model_inputs(
    data: dict[str, Any],
    *,
    model_input_fields: list[str],
) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key in model_input_fields:
        if key in data:
            value = data[key]
            if torch.is_tensor(value):
                filtered[key] = value.clone().detach()
            else:
                filtered[key] = copy.deepcopy(value)
    return filtered


def load_atoms(structure_path: str) -> Atoms:
    atoms = read(structure_path)
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected an ASE Atoms object from `{structure_path}`.")
    return atoms


def resolve_nequip_model_path(model_name: str) -> Path:
    from mattertune.backbones.nequip_foundation.nequip_model import CACHE_DIR
    from mattertune.backbones.nequip_foundation.nequip_model import MODEL_URLS

    if model_name not in MODEL_URLS:
        supported = ", ".join(sorted(MODEL_URLS))
        raise ValueError(
            f"Unknown NequIP packaged model `{model_name}`. Supported values are: {supported}."
        )

    package_path = CACHE_DIR / f"{model_name}.nequip.zip"
    if not package_path.exists():
        print(f"Downloading {model_name} to {package_path}")
        torch.hub.download_url_to_file(MODEL_URLS[model_name], str(package_path))
    return package_path


def load_model_components(model_name: str, device: torch.device):
    from nequip.integrations.utils import basic_transforms
    from nequip.model.saved_models.package import ModelFromPackage
    from nequip.nn.graph_model import R_MAX_KEY
    from nequip.nn.graph_model import TYPE_NAMES_KEY

    package_path = resolve_nequip_model_path(model_name)
    package = ModelFromPackage(str(package_path))
    graph_model = package["sole_model"].to(device).eval()

    if not hasattr(graph_model, "metadata"):
        raise TypeError("Expected a packaged NequIP GraphModel with metadata.")

    metadata = graph_model.metadata
    r_max = float(metadata[R_MAX_KEY])
    type_names = metadata[TYPE_NAMES_KEY].split(" ")

    transforms = basic_transforms(
        metadata=metadata,
        r_max=r_max,
        type_names=type_names,
        chemical_species_to_atom_type_map={sym: sym for sym in type_names},
    )
    if len(transforms) != 2:
        raise RuntimeError(
            f"Expected exactly two basic transforms (atom type + neighbor list), got {len(transforms)}."
        )
    atomtype_transform, neighbor_transform = transforms

    force_module = graph_model.model
    if not hasattr(force_module, "func"):
        raise TypeError(
            "Expected the packaged NequIP model to expose an energy model as `graph_model.model.func`."
        )
    energy_model = force_module.func
    energy_model.eval()

    return (
        graph_model,
        energy_model,
        atomtype_transform,
        neighbor_transform,
        list(graph_model.model_input_fields),
        package_path,
    )


def prepare_input(
    atoms: Atoms,
    *,
    atomtype_transform,
    neighbor_transform,
    device: torch.device,
):
    from nequip.data import AtomicDataDict
    from nequip.data.ase import from_ase

    data = from_ase(atoms)
    data = atomtype_transform(data)
    data = neighbor_transform(data)
    data = AtomicDataDict.to_(data, device)  # type: ignore[assignment]
    return data


def run_standard_energy_force(
    energy_model,
    data: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], TimingRecord]:
    from nequip.data import AtomicDataDict

    prepared = clone_atomic_data(data)
    pos = prepared[AtomicDataDict.POSITIONS_KEY]
    pos.requires_grad_(True)
    prepared[AtomicDataDict.POSITIONS_KEY] = pos

    sync_if_needed(device)
    t_forward_start = time.perf_counter()
    output = energy_model(prepared)
    sync_if_needed(device)
    t_forward_end = time.perf_counter()

    total_energy = output[AtomicDataDict.TOTAL_ENERGY_KEY]
    sync_if_needed(device)
    t_grad_start = time.perf_counter()
    (grad,) = torch.autograd.grad(
        total_energy.sum(),
        pos,
        create_graph=False,
    )
    sync_if_needed(device)
    t_grad_end = time.perf_counter()

    output = dict(output)
    output[AtomicDataDict.FORCE_KEY] = -grad

    return output, TimingRecord(
        prepare_ms=0.0,
        backbone_ms=(t_forward_end - t_forward_start) * 1000.0,
        autograd_ms=(t_grad_end - t_grad_start) * 1000.0,
        total_ms=(t_grad_end - t_forward_start) * 1000.0,
    )


def run_partial_energy_force(
    energy_model,
    data: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], TimingRecord]:
    from nequip.data import AtomicDataDict

    prepared = clone_atomic_data(data)
    pos = prepared[AtomicDataDict.POSITIONS_KEY]
    pos.requires_grad_(True)
    prepared[AtomicDataDict.POSITIONS_KEY] = pos

    sync_if_needed(device)
    t_forward_start = time.perf_counter()
    output = energy_model(prepared)
    sync_if_needed(device)
    t_forward_end = time.perf_counter()

    per_atom_energy = output[AtomicDataDict.PER_ATOM_ENERGY_KEY].view(-1)
    partial_forces: list[torch.Tensor] = []

    sync_if_needed(device)
    t_grad_start = time.perf_counter()
    num_atoms = int(per_atom_energy.shape[0])
    for atom_i in range(num_atoms):
        (grad_i,) = torch.autograd.grad(
            per_atom_energy[atom_i],
            pos,
            retain_graph=atom_i < num_atoms - 1,
            create_graph=False,
        )
        partial_forces.append(-grad_i)
    partial_force_tensor = torch.stack(partial_forces, dim=0)
    sync_if_needed(device)
    t_grad_end = time.perf_counter()

    output = dict(output)
    output[AtomicDataDict.PARTIAL_FORCE_KEY] = partial_force_tensor
    output["forces_from_partial"] = partial_force_tensor.sum(dim=0)

    return output, TimingRecord(
        prepare_ms=0.0,
        backbone_ms=(t_forward_end - t_forward_start) * 1000.0,
        autograd_ms=(t_grad_end - t_grad_start) * 1000.0,
        total_ms=(t_grad_end - t_forward_start) * 1000.0,
    )


def benchmark_standard_iteration(
    atoms: Atoms,
    *,
    atomtype_transform,
    neighbor_transform,
    model_input_fields: list[str],
    energy_model,
    device: torch.device,
) -> tuple[TimingRecord, dict[str, torch.Tensor]]:
    sync_if_needed(device)
    t_prepare_start = time.perf_counter()
    prepared = prepare_input(
        atoms,
        atomtype_transform=atomtype_transform,
        neighbor_transform=neighbor_transform,
        device=device,
    )
    prepared = filter_model_inputs(prepared, model_input_fields=model_input_fields)
    sync_if_needed(device)
    t_prepare_end = time.perf_counter()

    output, timing = run_standard_energy_force(
        energy_model,
        prepared,
        device=device,
    )
    timing.prepare_ms = (t_prepare_end - t_prepare_start) * 1000.0
    timing.total_ms += timing.prepare_ms
    return timing, output


def benchmark_partial_iteration(
    atoms: Atoms,
    *,
    atomtype_transform,
    neighbor_transform,
    model_input_fields: list[str],
    energy_model,
    device: torch.device,
) -> tuple[TimingRecord, dict[str, torch.Tensor]]:
    sync_if_needed(device)
    t_prepare_start = time.perf_counter()
    prepared = prepare_input(
        atoms,
        atomtype_transform=atomtype_transform,
        neighbor_transform=neighbor_transform,
        device=device,
    )
    prepared = filter_model_inputs(prepared, model_input_fields=model_input_fields)
    sync_if_needed(device)
    t_prepare_end = time.perf_counter()

    output, timing = run_partial_energy_force(
        energy_model,
        prepared,
        device=device,
    )
    timing.prepare_ms = (t_prepare_end - t_prepare_start) * 1000.0
    timing.total_ms += timing.prepare_ms
    return timing, output


def summarize(records: list[TimingRecord]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in ("prepare_ms", "backbone_ms", "autograd_ms", "total_ms"):
        values = [getattr(record, field) for record in records]
        summary[field] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def print_summary(title: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    for key, stats in summary.items():
        print(
            f"  {key:>12s}: mean={stats['mean']:10.4f} ms  "
            f"std={stats['stdev']:10.4f}  min={stats['min']:10.4f}  max={stats['max']:10.4f}"
        )


def to_numpy_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(-1)[0].item())


def validate_reference_outputs(
    atoms: Atoms,
    *,
    graph_model,
    atomtype_transform,
    neighbor_transform,
    model_input_fields: list[str],
    energy_model,
    device: torch.device,
    internal_energy_atol: float,
    internal_force_atol: float,
    graph_energy_atol: float,
    graph_force_atol: float,
) -> dict[str, float]:
    from nequip.data import AtomicDataDict

    prepared = prepare_input(
        atoms,
        atomtype_transform=atomtype_transform,
        neighbor_transform=neighbor_transform,
        device=device,
    )
    prepared_for_manual = filter_model_inputs(
        prepared,
        model_input_fields=model_input_fields,
    )

    prepared_internal = clone_atomic_data(prepared_for_manual)
    pos = prepared_internal[AtomicDataDict.POSITIONS_KEY]
    pos.requires_grad_(True)
    prepared_internal[AtomicDataDict.POSITIONS_KEY] = pos
    internal_output = energy_model(prepared_internal)
    internal_total_energy = internal_output[AtomicDataDict.TOTAL_ENERGY_KEY]
    internal_per_atom_energy = internal_output[AtomicDataDict.PER_ATOM_ENERGY_KEY].view(-1)

    (internal_force,) = torch.autograd.grad(
        internal_total_energy.sum(),
        pos,
        retain_graph=True,
        create_graph=False,
    )
    internal_force = -internal_force

    partial_forces: list[torch.Tensor] = []
    num_atoms = int(internal_per_atom_energy.shape[0])
    for atom_i in range(num_atoms):
        (grad_i,) = torch.autograd.grad(
            internal_per_atom_energy[atom_i],
            pos,
            retain_graph=atom_i < num_atoms - 1,
            create_graph=False,
        )
        partial_forces.append(-grad_i)
    internal_partial_force = torch.stack(partial_forces, dim=0)

    sync_if_needed(device)
    reference = graph_model(clone_atomic_data(prepared))
    sync_if_needed(device)

    standard_output, _ = run_standard_energy_force(
        energy_model,
        prepared_for_manual,
        device=device,
    )

    total_energy_ref = reference[AtomicDataDict.TOTAL_ENERGY_KEY].detach().cpu()
    forces_ref = reference[AtomicDataDict.FORCE_KEY].detach().cpu()
    total_energy_std = standard_output[AtomicDataDict.TOTAL_ENERGY_KEY].detach().cpu()
    forces_std = standard_output[AtomicDataDict.FORCE_KEY].detach().cpu()
    internal_total_energy = internal_total_energy.detach().cpu()
    internal_per_atom_energy = internal_per_atom_energy.detach().cpu()
    internal_force = internal_force.detach().cpu()
    internal_partial_force = internal_partial_force.detach().cpu()
    internal_energy_abs_diff = float(
        (internal_per_atom_energy.sum().view(-1) - internal_total_energy.view(-1))
        .abs()
        .max()
        .item()
    )
    internal_force_abs_diff = float(
        (internal_partial_force.sum(dim=0) - internal_force).abs().max().item()
    )

    graph_energy_abs_diff = float(
        (total_energy_ref.view(-1) - total_energy_std.view(-1)).abs().max().item()
    )
    graph_force_abs_diff = float((forces_ref - forces_std).abs().max().item())
    if graph_energy_abs_diff > graph_energy_atol:
        raise RuntimeError(
            "Manual standard energy deviates too much from the packaged model output: "
            f"{graph_energy_abs_diff:.6e} eV > {graph_energy_atol:.6e} eV."
        )
    if graph_force_abs_diff > graph_force_atol:
        raise RuntimeError(
            "Manual standard forces deviate too much from the packaged model output: "
            f"{graph_force_abs_diff:.6e} eV/A > {graph_force_atol:.6e} eV/A."
        )
    if internal_energy_abs_diff > internal_energy_atol:
        raise RuntimeError(
            "Per-atom energies do not sum to the total energy closely enough: "
            f"{internal_energy_abs_diff:.6e} eV > {internal_energy_atol:.6e} eV."
        )
    if internal_force_abs_diff > internal_force_atol:
        raise RuntimeError(
            "Summed partial forces do not match standard forces closely enough: "
            f"{internal_force_abs_diff:.6e} eV/A > {internal_force_atol:.6e} eV/A."
        )
    return {
        "internal_energy_sum_abs_diff_eV": internal_energy_abs_diff,
        "internal_force_sum_abs_diff_eVA": internal_force_abs_diff,
        "graph_energy_abs_diff_eV": graph_energy_abs_diff,
        "graph_force_abs_diff_eVA": graph_force_abs_diff,
    }


def main(args: argparse.Namespace) -> None:
    from nequip.data import AtomicDataDict

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device(args.device)
    atoms = load_atoms(args.structure)

    (
        graph_model,
        energy_model,
        atomtype_transform,
        neighbor_transform,
        model_input_fields,
        package_path,
    ) = load_model_components(args.model_name, device)

    print(f"Loaded packaged model: {args.model_name}")
    print(f"Package path: {package_path}")
    print(f"Device: {device}")
    print(f"Structure: {args.structure}")
    print(f"Num atoms: {len(atoms)}")
    print(f"Cell rank: {atoms.cell.rank}")
    print(
        "Benchmark modes:\n"
        "  standard = total energy + force from one autograd.grad(total_energy, pos)\n"
        "  partial  = per-atom energy + looped autograd.grad(e_i, pos) for all atoms"
    )

    reference_diffs = validate_reference_outputs(
        atoms,
        graph_model=graph_model,
        atomtype_transform=atomtype_transform,
        neighbor_transform=neighbor_transform,
        model_input_fields=model_input_fields,
        energy_model=energy_model,
        device=device,
        internal_energy_atol=args.internal_energy_atol,
        internal_force_atol=args.internal_force_atol,
        graph_energy_atol=args.graph_energy_atol,
        graph_force_atol=args.graph_force_atol,
    )
    print(
        "Correctness checks passed. "
        f"Internal deltas: "
        f"d(sum e_i, E)={reference_diffs['internal_energy_sum_abs_diff_eV']:.6e} eV, "
        f"max|d(sum_i f_i, F)|={reference_diffs['internal_force_sum_abs_diff_eVA']:.6e} eV/A. "
        f"Packaged-vs-manual deltas: "
        f"dE={reference_diffs['graph_energy_abs_diff_eV']:.6e} eV, "
        f"max|dF|={reference_diffs['graph_force_abs_diff_eVA']:.6e} eV/A"
    )

    print(f"Warmup iterations: {args.warmup}")
    for _ in range(args.warmup):
        benchmark_standard_iteration(
            atoms,
            atomtype_transform=atomtype_transform,
            neighbor_transform=neighbor_transform,
            model_input_fields=model_input_fields,
            energy_model=energy_model,
            device=device,
        )
        benchmark_partial_iteration(
            atoms,
            atomtype_transform=atomtype_transform,
            neighbor_transform=neighbor_transform,
            model_input_fields=model_input_fields,
            energy_model=energy_model,
            device=device,
        )

    standard_records: list[TimingRecord] = []
    partial_records: list[TimingRecord] = []
    iteration_rows: list[dict[str, float | int]] = []

    print(f"Recorded iterations: {args.repeats}")
    for iteration in range(args.repeats):
        standard_timing, standard_output = benchmark_standard_iteration(
            atoms,
            atomtype_transform=atomtype_transform,
            neighbor_transform=neighbor_transform,
            model_input_fields=model_input_fields,
            energy_model=energy_model,
            device=device,
        )
        partial_timing, partial_output = benchmark_partial_iteration(
            atoms,
            atomtype_transform=atomtype_transform,
            neighbor_transform=neighbor_transform,
            model_input_fields=model_input_fields,
            energy_model=energy_model,
            device=device,
        )

        standard_records.append(standard_timing)
        partial_records.append(partial_timing)

        row = {
            "iteration": iteration,
            "standard_prepare_ms": standard_timing.prepare_ms,
            "standard_backbone_ms": standard_timing.backbone_ms,
            "standard_autograd_ms": standard_timing.autograd_ms,
            "standard_total_ms": standard_timing.total_ms,
            "partial_prepare_ms": partial_timing.prepare_ms,
            "partial_backbone_ms": partial_timing.backbone_ms,
            "partial_autograd_ms": partial_timing.autograd_ms,
            "partial_total_ms": partial_timing.total_ms,
        }
        iteration_rows.append(row)

        print(
            f"iter={iteration:02d}  "
            f"std(total={standard_timing.total_ms:9.3f} ms, prep={standard_timing.prepare_ms:8.3f}, "
            f"backbone={standard_timing.backbone_ms:8.3f}, autograd={standard_timing.autograd_ms:8.3f})  "
            f"partial(total={partial_timing.total_ms:9.3f} ms, prep={partial_timing.prepare_ms:8.3f}, "
            f"backbone={partial_timing.backbone_ms:8.3f}, autograd={partial_timing.autograd_ms:8.3f})"
        )

    standard_summary = summarize(standard_records)
    partial_summary = summarize(partial_records)

    print_summary("Standard Energy+Force Timing", standard_summary)
    print_summary("Per-Atom Energy+Partial Force Timing", partial_summary)

    speedup = partial_summary["total_ms"]["mean"] / standard_summary["total_ms"]["mean"]
    print(f"\nPartial / standard mean total-time ratio: {speedup:.4f}x")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "iteration_times.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        header = list(iteration_rows[0].keys()) if iteration_rows else []
        handle.write(",".join(header) + "\n")
        for row in iteration_rows:
            handle.write(",".join(str(row[key]) for key in header) + "\n")

    summary_payload = {
        "model_name": args.model_name,
        "structure": str(Path(args.structure).resolve()),
        "device": str(device),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "internal_energy_atol": args.internal_energy_atol,
        "internal_force_atol": args.internal_force_atol,
        "graph_energy_atol": args.graph_energy_atol,
        "graph_force_atol": args.graph_force_atol,
        "packaged_vs_manual_reference": reference_diffs,
        "standard_summary_ms": standard_summary,
        "partial_summary_ms": partial_summary,
        "partial_over_standard_total_ratio": speedup,
        "reference_values": {
            "standard_total_energy_eV": to_numpy_scalar(
                standard_output[AtomicDataDict.TOTAL_ENERGY_KEY]
            ),
            "partial_total_energy_eV": to_numpy_scalar(
                partial_output[AtomicDataDict.TOTAL_ENERGY_KEY]
            ),
            "per_atom_energy_sum_eV": float(
                partial_output[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                .detach()
                .cpu()
                .view(-1)
                .sum()
                .item()
            ),
            "standard_max_force_eVA": float(
                standard_output[AtomicDataDict.FORCE_KEY].detach().cpu().abs().max().item()
            ),
            "partial_max_partial_force_eVA": float(
                partial_output[AtomicDataDict.PARTIAL_FORCE_KEY]
                .detach()
                .cpu()
                .abs()
                .max()
                .item()
            ),
        },
    }
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def parse_args() -> argparse.Namespace:
    default_structure = (
        Path(__file__).resolve().parents[1] / "electrolyte" / "data" / "LiH2O.xyz"
    )
    default_output_dir = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark NequIP-OAM standard energy/force inference versus "
            "per-atom-energy plus partial-force inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", default="NequIP-OAM-L-0.1")
    parser.add_argument("--structure", default=str(default_structure))
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--internal-energy-atol",
        type=float,
        default=1e-8,
        help="Tolerance for checking sum(per-atom energy)=total energy on a single forward graph.",
    )
    parser.add_argument(
        "--internal-force-atol",
        type=float,
        default=5e-3,
        help="Tolerance for checking sum(partial forces)=standard force on a single forward graph. GPU reductions are not bitwise exact here.",
    )
    parser.add_argument(
        "--graph-energy-atol",
        type=float,
        default=1e-2,
        help="Allowed absolute energy deviation between the manual energy-only path and the packaged NequIP graph model.",
    )
    parser.add_argument(
        "--graph-force-atol",
        type=float,
        default=1e-2,
        help="Allowed max absolute force deviation between the manual energy-only path and the packaged NequIP graph model.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(default_output_dir))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
