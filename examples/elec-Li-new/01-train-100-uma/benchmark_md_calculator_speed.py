from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import ase.units as units
import numpy as np
import torch
from ase.calculators.calculator import all_changes
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


EXPERIMENT_ROOT = Path(
    "/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/01-train-pair-100-uma"
)
DEFAULT_STRUCTURE = (
    "/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/get_all_trj_aimd/"
    "Li-metal/case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/top.pdb"
)


def find_latest_uma_checkpoint(force_mode: str | None) -> str:
    pattern = f"*/checkpoints/*-{force_mode}-pair100-best.ckpt" if force_mode else "*/checkpoints/*best.ckpt"
    candidates = sorted(
        EXPERIMENT_ROOT.glob(pattern),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        suffix = f" for force_mode={force_mode}" if force_mode else ""
        raise FileNotFoundError(f"No UMA finetuned checkpoint found under {EXPERIMENT_ROOT}{suffix}.")
    return str(candidates[-1])


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_cuda_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        else:
            torch.cuda.set_device(0)


def load_finetuned_calculator(family: str, checkpoint: str, device: str):
    from mattertune.main import load_finetuned_checkpoint

    model = load_finetuned_checkpoint(checkpoint, map_location="cpu")
    calc = model.ase_calculator(device=device)
    return calc, {
        "loader": "mattertune.load_finetuned_checkpoint().ase_calculator()",
        "checkpoint": checkpoint,
        "model_class": type(model).__name__,
        "implemented_properties": list(getattr(calc, "implemented_properties", [])),
        "hparams_name": getattr(getattr(model, "hparams", None), "name", family),
    }


def load_native_calculator(family: str, device: str, uma_merge_mole: bool):
    if family == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        calc = MatterSimCalculator(
            load_path="MatterSim-v1.0.0-1M",
            device=device,
            compute_stress=False,
        )
        return calc, {
            "loader": "mattersim.forcefield.MatterSimCalculator",
            "model_name": "MatterSim-v1.0.0-1M",
            "compute_stress": False,
            "implemented_properties": list(getattr(calc, "implemented_properties", [])),
        }

    if family == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        model, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            compile=False,
        )
        calc = ORBCalculator(model, atoms_adapter=atoms_adapter, device=device)
        return calc, {
            "loader": "orb_models.forcefield.pretrained.orb_v3_conservative_inf_omat + ORBCalculator",
            "model_name": "orb-v3-conservative-inf-omat",
            "implemented_properties": list(getattr(calc, "implemented_properties", [])),
        }

    if family == "uma":
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
        from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

        inference_settings: Any = "default"
        if uma_merge_mole:
            inference_settings = inference_settings_default()
            inference_settings.merge_mole = True
        calc = FAIRChemCalculator.from_model_checkpoint(
            "uma-s-1p1",
            task_name="omat",
            inference_settings=inference_settings,
            device="cuda" if device.startswith("cuda") else device,
            workers=1,
        )
        return calc, {
            "loader": "fairchem.core.calculate.ase_calculator.FAIRChemCalculator",
            "model_name": "uma-s-1p1",
            "task_name": "omat",
            "merge_mole": uma_merge_mole,
            "implemented_properties": list(getattr(calc, "implemented_properties", [])),
        }

    raise ValueError(f"Unsupported family: {family}")


def parse_supercell(raw_value: str) -> tuple[int, int, int]:
    pieces = [piece.strip() for piece in raw_value.split(",")]
    if len(pieces) != 3:
        raise ValueError("--supercell must have the form A,B,C, for example 2,2,2.")
    repeat = tuple(int(piece) for piece in pieces)
    if any(value <= 0 for value in repeat):
        raise ValueError("--supercell values must be positive integers.")
    return repeat


def make_atoms(
    structure: str,
    temperature: float,
    seed: int,
    supercell: tuple[int, int, int],
):
    atoms = read(structure)
    if supercell != (1, 1, 1):
        atoms = atoms.repeat(supercell)
    atoms.info.setdefault("task_name", "omat")
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=temperature,
        rng=np.random.default_rng(seed),
    )
    Stationary(atoms)
    return atoms


def benchmark_single_point(calc, atoms, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
        synchronize()

    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
        synchronize()
        times.append(time.perf_counter() - start)
    return summarize_times(times)


def benchmark_nvt_md(
    calc,
    *,
    structure: str,
    supercell: tuple[int, int, int],
    temperature: float,
    timestep_fs: float,
    friction_fs_inv: float,
    warmup_steps: int,
    steps: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    step_times: list[float] = []
    total_times: list[float] = []
    final_temperatures: list[float] = []

    for repeat_idx in range(repeats):
        atoms = make_atoms(structure, temperature, seed + repeat_idx, supercell)
        atoms.calc = calc
        _ = atoms.get_forces()
        synchronize()

        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature,
            friction=friction_fs_inv / units.fs,
            fixcm=False,
        )
        if warmup_steps > 0:
            dyn.run(warmup_steps)
            synchronize()

        start = time.perf_counter()
        dyn.run(steps)
        synchronize()
        elapsed = time.perf_counter() - start
        total_times.append(elapsed)
        step_times.append(elapsed / steps)
        final_temperatures.append(float(atoms.get_temperature()))

    summary = summarize_times(step_times)
    summary["total_times_s"] = total_times
    summary["final_temperatures_K"] = final_temperatures
    summary["steps"] = steps
    summary["warmup_steps"] = warmup_steps
    return summary


def summarize_times(times: list[float]) -> dict[str, Any]:
    ordered = sorted(times)
    return {
        "times_s": times,
        "mean_s": float(statistics.fmean(times)),
        "median_s": float(statistics.median(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "stdev_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        "p10_s": float(ordered[max(0, int(0.1 * (len(ordered) - 1)))]),
        "p90_s": float(ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("mattersim", "orb", "uma"), required=True)
    parser.add_argument("--mode", choices=("finetuned", "native"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force-mode", choices=("direct", "conservative"), default=None)
    parser.add_argument("--structure", default=DEFAULT_STRUCTURE)
    parser.add_argument("--supercell", default="1,1,1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--friction-fs-inv", type=float, default=0.02)
    parser.add_argument("--md-steps", type=int, default=50)
    parser.add_argument("--md-warmup-steps", type=int, default=5)
    parser.add_argument("--md-repeats", type=int, default=3)
    parser.add_argument("--skip-md", action="store_true")
    parser.add_argument("--sp-warmup", type=int, default=5)
    parser.add_argument("--sp-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--uma-merge-mole", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if not args.skip_md and args.md_steps <= 0:
        parser.error("--md-steps must be positive.")
    if (not args.skip_md and args.md_repeats <= 0) or args.sp_repeats <= 0:
        parser.error("repeat counts must be positive.")

    supercell = parse_supercell(args.supercell)
    set_cuda_device(args.device)
    torch.set_float32_matmul_precision("high")

    if args.mode == "finetuned":
        checkpoint = args.checkpoint
        if checkpoint is None:
            if args.family != "uma":
                parser.error("--checkpoint is required for non-UMA finetuned benchmarks in this UMA experiment directory.")
            checkpoint = find_latest_uma_checkpoint(args.force_mode)
        calc, calc_info = load_finetuned_calculator(args.family, checkpoint, args.device)
    else:
        checkpoint = None
        calc, calc_info = load_native_calculator(args.family, args.device, args.uma_merge_mole)

    atoms = make_atoms(args.structure, args.temperature, args.seed, supercell)
    natoms = len(atoms)
    formula = atoms.get_chemical_formula()

    # Load/first-use warmup is outside the measured regions.
    atoms.calc = calc
    _ = atoms.get_forces()
    synchronize()

    single_point = benchmark_single_point(
        calc,
        atoms,
        warmup=args.sp_warmup,
        repeats=args.sp_repeats,
    )
    md = None
    if not args.skip_md:
        md = benchmark_nvt_md(
            calc,
            structure=args.structure,
            supercell=supercell,
            temperature=args.temperature,
            timestep_fs=args.timestep_fs,
            friction_fs_inv=args.friction_fs_inv,
            warmup_steps=args.md_warmup_steps,
            steps=args.md_steps,
            repeats=args.md_repeats,
            seed=args.seed,
        )

    payload = {
        "family": args.family,
        "mode": args.mode,
        "force_mode": args.force_mode,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "structure": args.structure,
        "supercell": list(supercell),
        "natoms": natoms,
        "formula": formula,
        "temperature_K": args.temperature,
        "timestep_fs": args.timestep_fs,
        "friction_fs_inv": args.friction_fs_inv,
        "calculator": calc_info,
        "single_point_energy_forces": single_point,
        "nvt_md": md,
    }

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    del calc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
