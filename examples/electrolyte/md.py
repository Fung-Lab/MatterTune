from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import ase.units as units
import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from mattertune import load_pretrained_model
from mattertune.backbones import (
    EqV2BackboneModule,
    JMPBackboneModule,
    MACEBackboneModule,
    MatterSimM3GNetBackboneModule,
    M3GNetBackboneModule,
    NequIPBackboneModule,
    ORBBackboneModule,
    UMABackboneModule,
)
from mattertune.finetune.base import FinetuneModuleBase
from mattertune.pretrained import PretrainedModel
from mattertune.util import optional_import_error_message

from ghost_target_calculator import GhostTargetCorrectionCalculator


class TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_original = sys.stdout
        stderr_original = sys.stderr
        tee_stream = TeeStream(stdout_original, log_file)
        sys.stdout = tee_stream
        sys.stderr = tee_stream
        try:
            yield
        finally:
            sys.stdout = stdout_original
            sys.stderr = stderr_original


def lambda_mask_to_string(lambda_mask: np.ndarray) -> str:
    return np.array2string(
        np.asarray(lambda_mask, dtype=np.float64),
        precision=6,
        separator=", ",
        max_line_width=1_000_000,
    )


def annotate_frame_metadata(
    atoms: Atoms,
    *,
    step: int,
    time_fs: float,
    temperature: float,
    total_energy: float,
    lambda0_energy: float | None,
    lambda1_energy: float | None,
    lambda1_base_energy: float | None,
    ghost_lj_energy: float | None,
    lambda_array_name: str,
):
    atoms.info["md_step"] = int(step)
    atoms.info["time_fs"] = float(time_fs)
    atoms.info["temperature_K"] = float(temperature)
    atoms.info["total_energy_eV"] = float(total_energy)
    if lambda0_energy is not None:
        atoms.info["lambda0_energy_eV"] = float(lambda0_energy)
        atoms.info["initial_pes_energy_eV"] = float(lambda0_energy)
    if lambda1_energy is not None:
        atoms.info["lambda1_energy_eV"] = float(lambda1_energy)
        atoms.info["final_pes_energy_eV"] = float(lambda1_energy)
    if lambda1_base_energy is not None:
        atoms.info["lambda1_base_energy_eV"] = float(lambda1_base_energy)
        atoms.info["final_pes_energy_without_lj_eV"] = float(lambda1_base_energy)
    if ghost_lj_energy is not None:
        atoms.info["ghost_lj_energy_eV"] = float(ghost_lj_energy)
    atoms.arrays[lambda_array_name] = np.asarray(
        atoms.arrays[lambda_array_name], dtype=np.float64
    ).copy()


def format_step_log(
    *,
    step: int,
    time_fs: float,
    temperature: float,
    total_energy: float,
    lambda0_energy: float | None,
    lambda1_energy: float | None,
    lambda1_base_energy: float | None,
    ghost_lj_energy: float | None,
) -> str:
    lambda0_str = "nan" if lambda0_energy is None else f"{lambda0_energy: .12f}"
    lambda1_str = "nan" if lambda1_energy is None else f"{lambda1_energy: .12f}"
    lambda1_base_str = (
        "nan" if lambda1_base_energy is None else f"{lambda1_base_energy: .12f}"
    )
    ghost_lj_str = "nan" if ghost_lj_energy is None else f"{ghost_lj_energy: .12f}"
    return (
        f"step={step:5d} time_fs={time_fs:9.3f} "
        f"temp={temperature:8.3f} K "
        f"mixed_energy={total_energy: .12f} eV "
        f"lambda0_energy={lambda0_str} eV "
        f"lambda1_energy={lambda1_str} eV "
        f"lambda1_base={lambda1_base_str} eV "
        f"ghost_lj={ghost_lj_str} eV"
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def get_ghost_base_energy(
    *,
    lambda1_energy: float | None,
    ghost_lj_energy: float | None,
    calculator: GhostTargetCorrectionCalculator,
) -> float | None:
    details = calculator.last_ghost_endpoint_details
    if details is not None and details.get("base_energy_eV") is not None:
        return float(details["base_energy_eV"])
    if lambda1_energy is not None and ghost_lj_energy is not None:
        return float(lambda1_energy - ghost_lj_energy)
    return None


ENERGY_LOG_FIELDS = [
    "step",
    "time_fs",
    "time_ps",
    "temperature_K",
    "mixed_energy_eV",
    "E_I_eV",
    "E_F_with_LJ_eV",
    "E_F_without_LJ_eV",
    "E_LJ_eV",
    "deltaE_with_LJ_eV",
    "deltaE_without_LJ_eV",
]


def write_energy_log_row(
    writer: csv.DictWriter,
    *,
    step: int,
    time_fs: float,
    temperature: float,
    total_energy: float,
    lambda0_energy: float | None,
    lambda1_energy: float | None,
    lambda1_base_energy: float | None,
    ghost_lj_energy: float | None,
) -> None:
    delta_with_lj = (
        None
        if lambda0_energy is None or lambda1_energy is None
        else lambda1_energy - lambda0_energy
    )
    delta_without_lj = (
        None
        if lambda0_energy is None or lambda1_base_energy is None
        else lambda1_base_energy - lambda0_energy
    )
    writer.writerow(
        {
            "step": int(step),
            "time_fs": float(time_fs),
            "time_ps": float(time_fs / 1000.0),
            "temperature_K": float(temperature),
            "mixed_energy_eV": float(total_energy),
            "E_I_eV": _optional_float(lambda0_energy),
            "E_F_with_LJ_eV": _optional_float(lambda1_energy),
            "E_F_without_LJ_eV": _optional_float(lambda1_base_energy),
            "E_LJ_eV": _optional_float(ghost_lj_energy),
            "deltaE_with_LJ_eV": _optional_float(delta_with_lj),
            "deltaE_without_LJ_eV": _optional_float(delta_without_lj),
        }
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_diagnostics_record(
    diagnostics_path: Path | None,
    *,
    model_family: str,
    model_name: str,
    ghost_endpoint_mode: str,
    step: int,
    time_fs: float,
    lambda_mask: np.ndarray,
    total_energy: float,
    lambda0_energy: float | None,
    lambda1_energy: float | None,
    calculator: GhostTargetCorrectionCalculator,
):
    if diagnostics_path is None:
        return

    payload = {
        "model_family": model_family,
        "model_name": model_name,
        "ghost_endpoint_mode": ghost_endpoint_mode,
        "step": int(step),
        "time_fs": float(time_fs),
        "lambda_mask": np.asarray(lambda_mask, dtype=np.float64),
        "total_energy_eV": float(total_energy),
        "lambda0_energy_eV": None if lambda0_energy is None else float(lambda0_energy),
        "lambda1_energy_eV": None if lambda1_energy is None else float(lambda1_energy),
        "endpoint_gap_eV": (
            None
            if lambda0_energy is None or lambda1_energy is None
            else float(lambda1_energy - lambda0_energy)
        ),
        "real_endpoint": calculator.last_real_endpoint_details,
        "ghost_endpoint": calculator.last_ghost_endpoint_details,
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(payload), ensure_ascii=True) + "\n")


def build_default_atoms() -> Atoms:
    atoms = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
    atoms.pbc = True
    return atoms


def load_atoms(structure_path: str | None) -> Atoms:
    if structure_path is None:
        return build_default_atoms()

    atoms = read(structure_path)
    if not isinstance(atoms, Atoms):
        raise TypeError(
            f"Expected an ASE Atoms object from `{structure_path}`.")
    return atoms


def parse_target_indices(raw_value: str | None, natoms: int) -> list[int]:
    if raw_value is None:
        return [0]
    indices = [int(piece) for piece in raw_value.split(",") if piece.strip()]
    if not indices:
        raise ValueError("At least one target index must be provided.")
    for index in indices:
        if index < 0 or index >= natoms:
            raise ValueError(
                f"Target index {index} is out of bounds for a structure with {natoms} atoms."
            )
    return sorted(dict.fromkeys(indices))


def build_target_mask(natoms: int, target_indices: list[int]) -> np.ndarray:
    target_mask = np.zeros(natoms, dtype=bool)
    target_mask[target_indices] = True
    return target_mask


def build_lambda_mask(
    natoms: int,
    target_indices: list[int],
    lambda_value: float,
) -> np.ndarray:
    if lambda_value < 0.0 or lambda_value > 1.0:
        raise ValueError("lambda_value must lie in [0, 1].")
    lambda_mask = np.zeros(natoms, dtype=np.float64)
    lambda_mask[target_indices] = lambda_value
    return lambda_mask


def should_enable_uma_merge_experts(args: argparse.Namespace) -> bool:
    return (
        args.model_type is not None
        and args.model_type.strip().lower() == "uma"
        and args.uma_merge_experts
    )


class ASECalculatorBackedModel:
    def __init__(
        self,
        *,
        family: str,
        model_name: str,
        device: str,
        calculator,
    ):
        self.family = family
        self.model_name = model_name
        self.device = device
        self._calculator = calculator
        self.implemented_properties = tuple(
            getattr(calculator, "implemented_properties", [])
        )

    def supports_dummy_endpoint(self) -> bool:
        return False

    def predict_one(
        self,
        atoms: Atoms,
        properties: list[str] | None = None,
    ) -> dict[str, object]:
        requested = list(
            self.implemented_properties) if properties is None else properties
        atoms_copy = copy.deepcopy(atoms)
        self._calculator.calculate(
            atoms_copy,
            properties=requested,
            system_changes=["positions", "numbers", "cell",
                            "pbc", "initial_charges", "initial_magmoms"],
        )
        return copy.deepcopy(self._calculator.results)


def _checkpoint_module_class(backbone_name: str):
    mapping = {
        "mattersim": MatterSimM3GNetBackboneModule,
        "orb": ORBBackboneModule,
        "mace": MACEBackboneModule,
        "uma": UMABackboneModule,
        "eqv2": EqV2BackboneModule,
        "jmp": JMPBackboneModule,
        "m3gnet": M3GNetBackboneModule,
        "nequip": NequIPBackboneModule,
        "allegro": NequIPBackboneModule,
    }
    try:
        return mapping[backbone_name]
    except KeyError as exc:
        supported = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unsupported finetuned checkpoint backbone `{backbone_name}`. "
            f"Supported values are: {supported}."
        ) from exc


def load_finetuned_model_from_checkpoint(
    ckpt_path: str,
    *,
    device: str,
):
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    backbone_name = hparams.get("name")
    if backbone_name is None:
        raise ValueError(
            f"Could not determine backbone name from checkpoint `{ckpt_path}`."
        )

    module_cls = _checkpoint_module_class(str(backbone_name))
    module: FinetuneModuleBase = module_cls.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location="cpu",
    )
    calculator = module.ase_calculator(device=device)
    return ASECalculatorBackedModel(
        family=str(backbone_name),
        model_name=Path(ckpt_path).stem,
        device=device,
        calculator=calculator,
    )


def load_md_models(
    args: argparse.Namespace,
) -> tuple[PretrainedModel | ASECalculatorBackedModel, PretrainedModel | ASECalculatorBackedModel | None]:
    if args.ckpt_path is not None:
        model = load_finetuned_model_from_checkpoint(
            args.ckpt_path,
            device=args.device,
        )
        return model, None

    load_kwargs: dict[str, object] = {}
    if args.task_name is not None:
        load_kwargs["task_name"] = args.task_name

    if should_enable_uma_merge_experts(args):
        from fairchem.core.units.mlip_unit.api.inference import (
            inference_settings_default,
        )

        inference_settings = inference_settings_default()
        inference_settings.merge_mole = True
        load_kwargs["inference_settings"] = inference_settings

    model = load_pretrained_model(
        args.model_type,
        args.model_name,
        device=args.device,
        **load_kwargs,
    )

    ghost_model = None
    if should_enable_uma_merge_experts(args) and 1e-8 < args.lambda_value < 1.0 - 1e-8:
        ghost_model = load_pretrained_model(
            args.model_type,
            args.model_name,
            device=args.device,
            **load_kwargs,
        )

    return model, ghost_model


def main(args: argparse.Namespace) -> None:
    atoms = load_atoms(args.structure)
    target_indices = parse_target_indices(args.target_indices, len(atoms))
    target_mask = build_target_mask(len(atoms), target_indices)
    lambda_mask = build_lambda_mask(
        len(atoms), target_indices, args.lambda_value)
    atoms.arrays[args.lambda_array_name] = lambda_mask
    atoms.arrays[args.target_array_name] = target_mask

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / args.trajectory_name
    final_structure_path = output_dir / args.final_structure_name
    log_path = trajectory_path.with_suffix(".txt")
    diagnostics_path = (
        None
        if args.diagnostics_name is None
        else output_dir / args.diagnostics_name
    )
    energy_log_path = (
        None
        if args.energy_log_name is None
        else output_dir / args.energy_log_name
    )

    with tee_output(log_path):
        model, ghost_model = load_md_models(args)
        print(
            f"model={model.family}:{model.model_name} device={model.device} "
            f"target_indices={target_indices} lambda={args.lambda_value:.6f} "
            f"ghost_mode={args.ghost_endpoint_mode} use_d3={args.use_d3}"
        )

        calc = GhostTargetCorrectionCalculator(
            model,
            ghost_model=ghost_model,
            lambda_array_name=args.lambda_array_name,
            target_array_name=args.target_array_name,
            epsilon=args.epsilon,
            sigma=args.sigma,
            alpha=args.alpha,
            rc=args.rc,
            ro=args.ro,
            smooth=args.smooth,
            ghost_endpoint_mode=args.ghost_endpoint_mode,
            use_d3=args.use_d3,
            d3_method=args.d3_method,
            d3_damping=args.d3_damping,
        )
        atoms.calc = calc

        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()

        if args.init_velocities:
            MaxwellBoltzmannDistribution(
                atoms,
                temperature_K=args.temperature,
                rng=np.random.default_rng(args.seed),
            )
            Stationary(atoms)
            print(
                f"initialized velocities at {args.temperature:.3f} K with seed={args.seed}"
            )
        else:
            print("initial velocities not assigned")

        dynamics = Langevin(
            atoms,
            timestep=args.timestep_fs * units.fs,
            temperature_K=args.temperature,
            friction=args.friction_fs_inv / units.fs,
            fixcm=False,
        )

        if trajectory_path.exists():
            trajectory_path.unlink()
        if diagnostics_path is not None and diagnostics_path.exists():
            diagnostics_path.unlink()
        if energy_log_path is not None and energy_log_path.exists():
            energy_log_path.unlink()

        step_cache: dict[str, object] = {"step": None, "values": None}

        def collect_step_values() -> dict[str, float | int | None]:
            step = dynamics.nsteps
            if step_cache["step"] == step and step_cache["values"] is not None:
                return step_cache["values"]  # type: ignore[return-value]

            time_fs = step * args.timestep_fs
            energy = atoms.get_potential_energy()
            temperature = atoms.get_temperature()
            lambda0_energy = calc.last_real_endpoint_energy
            lambda1_energy = calc.last_ghost_endpoint_energy
            ghost_lj_energy = calc.last_ghost_endpoint_lj_energy
            lambda1_base_energy = get_ghost_base_energy(
                lambda1_energy=lambda1_energy,
                ghost_lj_energy=ghost_lj_energy,
                calculator=calc,
            )

            values: dict[str, float | int | None] = {
                "step": int(step),
                "time_fs": float(time_fs),
                "temperature": float(temperature),
                "total_energy": float(energy),
                "lambda0_energy": lambda0_energy,
                "lambda1_energy": lambda1_energy,
                "lambda1_base_energy": lambda1_base_energy,
                "ghost_lj_energy": ghost_lj_energy,
            }
            step_cache["step"] = step
            step_cache["values"] = values
            return values

        energy_log_handle = None
        energy_log_writer = None
        if energy_log_path is not None:
            energy_log_path.parent.mkdir(parents=True, exist_ok=True)
            energy_log_handle = energy_log_path.open(
                "w", encoding="utf-8", newline=""
            )
            energy_log_writer = csv.DictWriter(
                energy_log_handle,
                fieldnames=ENERGY_LOG_FIELDS,
            )
            energy_log_writer.writeheader()

        def record_trajectory_step() -> None:
            values = collect_step_values()
            frame = atoms.copy()
            annotate_frame_metadata(
                frame,
                step=int(values["step"]),
                time_fs=float(values["time_fs"]),
                temperature=float(values["temperature"]),
                total_energy=float(values["total_energy"]),
                lambda0_energy=_optional_float(values["lambda0_energy"]),
                lambda1_energy=_optional_float(values["lambda1_energy"]),
                lambda1_base_energy=_optional_float(values["lambda1_base_energy"]),
                ghost_lj_energy=_optional_float(values["ghost_lj_energy"]),
                lambda_array_name=args.lambda_array_name,
            )
            write(
                trajectory_path,
                frame,
                append=True,
                format="extxyz",
            )

        def record_log_step() -> None:
            values = collect_step_values()
            print(
                format_step_log(
                    step=int(values["step"]),
                    time_fs=float(values["time_fs"]),
                    temperature=float(values["temperature"]),
                    total_energy=float(values["total_energy"]),
                    lambda0_energy=_optional_float(values["lambda0_energy"]),
                    lambda1_energy=_optional_float(values["lambda1_energy"]),
                    lambda1_base_energy=_optional_float(values["lambda1_base_energy"]),
                    ghost_lj_energy=_optional_float(values["ghost_lj_energy"]),
                )
            )
            if energy_log_writer is not None:
                write_energy_log_row(
                    energy_log_writer,
                    step=int(values["step"]),
                    time_fs=float(values["time_fs"]),
                    temperature=float(values["temperature"]),
                    total_energy=float(values["total_energy"]),
                    lambda0_energy=_optional_float(values["lambda0_energy"]),
                    lambda1_energy=_optional_float(values["lambda1_energy"]),
                    lambda1_base_energy=_optional_float(values["lambda1_base_energy"]),
                    ghost_lj_energy=_optional_float(values["ghost_lj_energy"]),
                )
                assert energy_log_handle is not None
                energy_log_handle.flush()

        def record_diagnostics_step() -> None:
            values = collect_step_values()
            write_diagnostics_record(
                diagnostics_path,
                model_family=model.family,
                model_name=model.model_name,
                ghost_endpoint_mode=args.ghost_endpoint_mode,
                step=int(values["step"]),
                time_fs=float(values["time_fs"]),
                lambda_mask=np.asarray(
                    atoms.arrays[args.lambda_array_name], dtype=np.float64
                ),
                total_energy=float(values["total_energy"]),
                lambda0_energy=_optional_float(values["lambda0_energy"]),
                lambda1_energy=_optional_float(values["lambda1_energy"]),
                calculator=calc,
            )

        dynamics.attach(record_log_step, interval=args.log_interval)
        dynamics.attach(record_trajectory_step, interval=args.trajectory_interval)
        if diagnostics_path is not None:
            dynamics.attach(record_diagnostics_step, interval=args.diagnostics_interval)

        print(
            f"running ghost-target MD: steps={args.steps}, timestep_fs={args.timestep_fs}, "
            f"temperature_K={args.temperature}, log_interval={args.log_interval}, "
            f"trajectory_interval={args.trajectory_interval}, "
            f"diagnostics_interval={args.diagnostics_interval}"
        )

        try:
            start_time = time.time()
            dynamics.run(args.steps)
            end_time = time.time()
        finally:
            if energy_log_handle is not None:
                energy_log_handle.close()
        print(f"MD simulation completed in {end_time - start_time:.2f} seconds")
        if args.steps > 0:
            print(
                f"Average MD step time: {(end_time - start_time) / args.steps:.2f} seconds"
            )

        final_energy = atoms.get_potential_energy()
        final_temperature = atoms.get_temperature()
        final_frame = atoms.copy()
        annotate_frame_metadata(
            final_frame,
            step=dynamics.nsteps,
            time_fs=dynamics.nsteps * args.timestep_fs,
            temperature=final_temperature,
            total_energy=final_energy,
            lambda0_energy=calc.last_real_endpoint_energy,
            lambda1_energy=calc.last_ghost_endpoint_energy,
            lambda1_base_energy=get_ghost_base_energy(
                lambda1_energy=calc.last_ghost_endpoint_energy,
                ghost_lj_energy=calc.last_ghost_endpoint_lj_energy,
                calculator=calc,
            ),
            ghost_lj_energy=calc.last_ghost_endpoint_lj_energy,
            lambda_array_name=args.lambda_array_name,
        )
        write(final_structure_path, final_frame)
        print(f"trajectory written to {trajectory_path}")
        print(f"log written to {log_path}")
        if energy_log_path is not None:
            print(f"energy log written to {energy_log_path}")
        if diagnostics_path is not None:
            print(f"diagnostics written to {diagnostics_path}")
        print(f"final structure written to {final_structure_path}")


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parent / "outputs" / "md"
    parser = argparse.ArgumentParser(
        description="Run MD with a pretrained model plus ghost-target excluded-volume correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Optional MatterTune finetuned checkpoint path. If provided, md.py loads this checkpoint instead of a pretrained model.",
    )
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--no-uma-merge-experts",
        action="store_false",
        dest="uma_merge_experts",
        help="Disable UMA MOE expert merging during MD. By default it is enabled for UMA models.",
    )
    parser.add_argument(
        "--use-d3",
        action="store_true",
        help="Add a Python-package DFT-D3 dispersion correction on top of the ghost-target calculator.",
    )
    parser.add_argument(
        "--d3-method",
        default="pbe",
        help="Method label passed to the Python DFTD3 calculator, for example `pbe`.",
    )
    parser.add_argument(
        "--d3-damping",
        default="d3bj",
        help="Damping mode passed to the Python DFTD3 calculator, for example `d3bj`.",
    )
    parser.add_argument("--structure", default=None)
    parser.add_argument(
        "--target-indices",
        default=None,
        help="Comma-separated alchemical target atom indices. Defaults to 0.",
    )
    parser.add_argument("--lambda-array-name", default="alchemical_lambda")
    parser.add_argument("--target-array-name", default="alchemical_target")
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=1.0,
        help="Ghost fraction for the selected target atoms: 0 = fully real, 1 = fully ghost.",
    )
    parser.add_argument("--epsilon", type=float, default=0.00694)
    parser.add_argument("--sigma", type=float, default=2.337)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rc", type=float, default=3.0)
    parser.add_argument("--ro", type=float, default=1.5)
    parser.add_argument(
        "--no-smooth",
        action="store_false",
        dest="smooth",
        help="Disable the smooth cutoff for the ghost-target correction.",
    )
    parser.add_argument(
        "--ghost-endpoint-mode",
        choices=("delete", "dummy"),
        default="delete",
        help="How to construct the ghost endpoint. `dummy` currently supports pretrained MACE models.",
    )
    parser.set_defaults(smooth=True)
    parser.set_defaults(uma_merge_experts=True)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--init-velocities",
        action="store_true",
        help="Initialize Maxwell-Boltzmann velocities before MD. Disabled by default.",
    )
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--friction-fs-inv", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Interval, in MD steps, for stdout and compact energy CSV logging.",
    )
    parser.add_argument(
        "--trajectory-interval",
        type=int,
        default=None,
        help="Interval, in MD steps, for writing trajectory frames. Defaults to --log-interval.",
    )
    parser.add_argument(
        "--diagnostics-interval",
        type=int,
        default=None,
        help="Interval, in MD steps, for writing diagnostics JSONL records. Defaults to --log-interval.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--trajectory-name", default="ghost_md.xyz")
    parser.add_argument(
        "--energy-log-name",
        default="energy_log.csv",
        help="Compact CSV file for logged energies. Set to an empty string to disable.",
    )
    parser.add_argument("--final-structure-name", default="ghost_final.extxyz")
    parser.add_argument(
        "--diagnostics-name",
        default="",
        help="Optional JSONL file for per-step endpoint diagnostics. Set to an empty string to disable.",
    )
    args = parser.parse_args()
    if args.diagnostics_name == "":
        args.diagnostics_name = None
    if args.energy_log_name == "":
        args.energy_log_name = None
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive.")
    if args.trajectory_interval is None:
        args.trajectory_interval = args.log_interval
    if args.diagnostics_interval is None:
        args.diagnostics_interval = args.log_interval
    if args.trajectory_interval <= 0:
        parser.error("--trajectory-interval must be positive.")
    if args.diagnostics_interval <= 0:
        parser.error("--diagnostics-interval must be positive.")
    if args.ckpt_path is None and args.model_type is None:
        parser.error("Either --model-type or --ckpt-path must be provided.")
    return args


if __name__ == "__main__":
    main(parse_args())
