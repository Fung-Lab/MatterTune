from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from ase.calculators.calculator import all_changes
from ase.io import read


DEFAULT_STRUCTURES = {
    "BaCuO2_110_Y2O3_110": "/nethome/lkong88/workspace/FastMLIP/contents/BaCuO2_110_Y2O3_110.xyz",
    "BaO_100_CuO_100": "/nethome/lkong88/workspace/FastMLIP/contents/BaO_100_CuO_100.xyz",
    "CuO_100_Y2O3_100_orthorhombic": "/nethome/lkong88/workspace/FastMLIP/contents/CuO_100_Y2O3_100_orthorhombic.xyz",
}

UMA_CHECKPOINTS = {
    "uma-s-1": "/home/lkong88/.cache/fairchem/models--facebook--UMA/snapshots/38529caa2c51a9a8a0d71f0b56b79ac33bc9eceb/checkpoints/uma-s-1.pt",
    "uma-s-1p1": "/home/lkong88/.cache/fairchem/models--facebook--UMA/snapshots/be2896459a03fcde05e20d2fcefd11f450601fce/checkpoints/uma-s-1p1.pt",
    "uma-s-1p2": "/home/lkong88/.cache/fairchem/models--facebook--UMA/snapshots/9e0d80ebc07f0c777e14d53781e1a7dcb2fd8561/checkpoints/uma-s-1p2.pt",
}

UMA_SPECS = {
    "uma-s-1": {
        "active_params": "6.6M",
        "total_params": "150M",
        "note": "Original small checkpoint cached locally; not listed in current fairchem available_models.",
    },
    "uma-s-1p1": {
        "active_params": "6.6M",
        "total_params": "150M",
        "note": "Early small checkpoint listed by current fairchem.",
    },
    "uma-s-1p2": {
        "active_params": "6.6M",
        "total_params": "290M",
        "note": "Latest small checkpoint listed by current fairchem.",
    },
    "uma-m-1p1": {
        "active_params": "50M",
        "total_params": "1.4B",
        "note": "Medium checkpoint; not cached locally in this environment.",
    },
}


def set_cuda_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        else:
            torch.cuda.set_device(0)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(times: list[float]) -> dict[str, float | list[float]]:
    return {
        "times_s": times,
        "mean_s": float(statistics.fmean(times)),
        "median_s": float(statistics.median(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "stdev_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
    }


def build_calculator(checkpoint: str, device: str):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

    settings = inference_settings_default()
    settings.merge_mole = True
    fairchem_device = "cuda" if device.startswith("cuda") else device
    return FAIRChemCalculator.from_model_checkpoint(
        checkpoint,
        task_name="omat",
        inference_settings=settings,
        device=fairchem_device,
        workers=1,
    )


def benchmark_structure(calc: Any, structure_path: str, warmup: int, repeats: int) -> dict[str, Any]:
    atoms = read(structure_path)
    atoms.info.setdefault("task_name", "omat")
    calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
    synchronize()
    for _ in range(warmup):
        calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
        synchronize()

    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
        synchronize()
        times.append(time.perf_counter() - start)

    payload = summarize(times)
    payload["natoms"] = len(atoms)
    payload["formula"] = atoms.get_chemical_formula()
    payload["pbc"] = atoms.pbc.tolist()
    payload["cell_lengths_A"] = atoms.cell.lengths().tolist()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="uma-s-1,uma-s-1p1,uma-s-1p2")
    args = parser.parse_args()

    set_cuda_device(args.device)
    torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_name in [piece.strip() for piece in args.models.split(",") if piece.strip()]:
        checkpoint = UMA_CHECKPOINTS[model_name]
        ckpt_path = Path(checkpoint)
        model_info = {
            "model": model_name,
            "checkpoint": checkpoint,
            "checkpoint_size_gib": ckpt_path.stat().st_size / 1024**3,
            **UMA_SPECS.get(model_name, {}),
        }
        for structure_name, structure_path in DEFAULT_STRUCTURES.items():
            print(f"LOAD {model_name} for {structure_name}", flush=True)
            calc = build_calculator(checkpoint, args.device)
            print(f"RUN {model_name} {structure_name}", flush=True)
            result = benchmark_structure(calc, structure_path, args.warmup, args.repeats)
            row = {
                **model_info,
                "structure": structure_name,
                "structure_path": structure_path,
                "device": args.device,
                "warmup": args.warmup,
                "repeats": args.repeats,
                **result,
            }
            rows.append(row)
            (output_dir / f"{model_name}_{structure_name}.json").write_text(
                json.dumps(row, indent=2),
                encoding="utf-8",
            )
            print(
                f"OK {model_name} {structure_name} natoms={row['natoms']} "
                f"median={row['median_s']:.3f}s",
                flush=True,
            )
            del calc
            torch.cuda.empty_cache()

    fields = [
        "model",
        "active_params",
        "total_params",
        "checkpoint_size_gib",
        "structure",
        "natoms",
        "median_s",
        "mean_s",
        "min_s",
        "max_s",
        "stdev_s",
        "device",
        "checkpoint",
        "note",
    ]
    with (output_dir / "combined_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# UMA pretrained version benchmark",
        "",
        f"Settings: native FAIRChemCalculator, task_name=omat, merge_mole=True, warmup={args.warmup}, repeats={args.repeats}, device={args.device}.",
        "",
        "| structure | atoms | model | active/total params | checkpoint | median (s) | mean (s) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['structure']} | {row['natoms']} | {row['model']} | "
            f"{row.get('active_params', '')}/{row.get('total_params', '')} | "
            f"{row['checkpoint_size_gib']:.2f} GiB | {row['median_s']:.3f} | {row['mean_s']:.3f} |"
        )
    (output_dir / "combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "combined_summary.md", flush=True)


if __name__ == "__main__":
    main()
