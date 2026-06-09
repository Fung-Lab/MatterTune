from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import all_changes
from ase.io import read


SCRIPT_DIR = Path(__file__).resolve().parent
MATTERTUNE_ROOT = SCRIPT_DIR.parents[2]
ELECTROLYTE_EXAMPLE_DIR = MATTERTUNE_ROOT / "examples" / "electrolyte"
DEFAULT_STRUCTURE = (
    "/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/get_all_trj_aimd/"
    "Li-metal/case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/top.pdb"
)
DEFAULT_CKPTS = {
    "mattersim": (
        "/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/"
        "03-train-pair-100/20260522-201116-mattersim-MatterSim-v1.0.0-1M-pair100-fw20-de05/"
        "checkpoints/mattersim-MatterSim-v1.0.0-1M-pair100-best.ckpt"
    ),
    "orb": (
        "/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/"
        "03-train-pair-100/20260523-175820-orb-orbv3-omat-conservative-inf-pair100-fw20-de05/"
        "checkpoints/orb-orb-v3-conservative-inf-omat-pair100-best.ckpt"
    ),
    "uma": (
        "/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/"
        "03-train-pair-100/20260524-225554-uma-uma-s1.1-pair100-fw20-de05/"
        "checkpoints/uma-uma-s-1p1-pair100-best.ckpt"
    ),
}


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def set_cuda_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(device.split(":", 1)[1]) if ":" in device else 0)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(times: list[float]) -> dict[str, Any]:
    return {
        "times_s": times,
        "mean_s": float(statistics.fmean(times)),
        "median_s": float(statistics.median(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "stdev_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
    }


def benchmark(label: str, fn, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
        synchronize()
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize()
        times.append(time.perf_counter() - start)
    result = summarize(times)
    result["label"] = label
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("mattersim", "orb", "uma"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--structure", default=DEFAULT_STRUCTURE)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ELECTROLYTE_EXAMPLE_DIR))
    md_module = _load_module("electrolyte_md_for_endpoint_benchmark", ELECTROLYTE_EXAMPLE_DIR / "md.py")
    ghost_module = _load_module(
        "ghost_target_calculator_for_endpoint_benchmark",
        ELECTROLYTE_EXAMPLE_DIR / "ghost_target_calculator.py",
    )
    GhostTargetCorrectionCalculator = ghost_module.GhostTargetCorrectionCalculator

    set_cuda_device(args.device)
    torch.set_float32_matmul_precision("high")

    checkpoint = args.checkpoint or DEFAULT_CKPTS[args.family]
    model = md_module.load_finetuned_model_from_checkpoint(checkpoint, device=args.device)

    atoms = read(args.structure)
    atoms.info.setdefault("task_name", "omat")
    if args.target_index < 0 or args.target_index >= len(atoms):
        raise ValueError(f"target-index out of range for natoms={len(atoms)}")
    target_mask = np.zeros(len(atoms), dtype=bool)
    target_mask[args.target_index] = True
    lambda_mask = np.zeros(len(atoms), dtype=np.float64)
    lambda_mask[args.target_index] = args.lambda_value
    atoms.arrays["alchemical_target"] = target_mask
    atoms.arrays["alchemical_lambda"] = lambda_mask

    lambda0_atoms = atoms.copy()
    lambda0_atoms.arrays["alchemical_lambda"] = np.zeros(len(atoms), dtype=np.float64)
    lambda0_atoms.arrays["alchemical_target"] = target_mask.copy()

    calc = GhostTargetCorrectionCalculator(
        model,
        lambda_array_name="alchemical_lambda",
        target_array_name="alchemical_target",
        epsilon=0.00694,
        sigma=2.337,
        smooth=True,
        ghost_endpoint_mode="delete",
        use_d3=False,
    )

    full_atoms = atoms.copy()
    reduced_atoms = atoms[np.flatnonzero(~target_mask)].copy()

    def real_endpoint_only():
        return calc._predict_endpoint(full_atoms, target_mask=target_mask, ghost_targets=False)

    def ghost_endpoint_only():
        return calc._predict_endpoint(full_atoms, target_mask=target_mask, ghost_targets=True)

    def direct_full_model_one_endpoint():
        return model.predict_one(full_atoms, properties=["energy", "forces"])

    def direct_reduced_model_one_endpoint():
        return model.predict_one(reduced_atoms, properties=["energy", "forces"])

    def current_two_endpoint_lambda_mid():
        calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
        return calc.results

    def current_two_endpoint_lambda_zero():
        calc.calculate(lambda0_atoms, properties=["energy", "forces"], system_changes=all_changes)
        return calc.results

    # First-use warmup outside all measured regions.
    current_two_endpoint_lambda_mid()
    synchronize()

    results = [
        benchmark("real_endpoint_only_internal", real_endpoint_only, args.warmup, args.repeats),
        benchmark("ghost_endpoint_only_internal", ghost_endpoint_only, args.warmup, args.repeats),
        benchmark("direct_full_model_one_endpoint", direct_full_model_one_endpoint, args.warmup, args.repeats),
        benchmark("direct_reduced_model_one_endpoint", direct_reduced_model_one_endpoint, args.warmup, args.repeats),
        benchmark("current_calculator_two_endpoints_lambda_0p5", current_two_endpoint_lambda_mid, args.warmup, args.repeats),
        benchmark("current_calculator_two_endpoints_lambda_0", current_two_endpoint_lambda_zero, args.warmup, args.repeats),
    ]

    by_label = {item["label"]: item for item in results}
    real = by_label["real_endpoint_only_internal"]["median_s"]
    ghost = by_label["ghost_endpoint_only_internal"]["median_s"]
    two = by_label["current_calculator_two_endpoints_lambda_0p5"]["median_s"]
    payload = {
        "family": args.family,
        "checkpoint": checkpoint,
        "structure": args.structure,
        "natoms": len(atoms),
        "reduced_natoms": len(reduced_atoms),
        "target_index": args.target_index,
        "lambda_value": args.lambda_value,
        "device": args.device,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "calculator_note": (
            "GhostTargetCorrectionCalculator.calculate currently evaluates both real and ghost "
            "endpoints before selecting/interpolating, so lambda=0 and lambda=0.5 both exercise "
            "the two-endpoint path."
        ),
        "results": results,
        "derived": {
            "two_endpoint_over_real_endpoint_median": two / real,
            "two_endpoint_over_ghost_endpoint_median": two / ghost,
            "two_endpoint_over_sum_real_plus_ghost_median": two / (real + ghost),
            "sum_real_plus_ghost_median_s": real + ghost,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "median_s", "mean_s", "min_s", "max_s", "stdev_s"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
