from __future__ import annotations

import argparse
import copy
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import ase.units as units
import numpy as np
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
    lambda_mask: np.ndarray,
    max_force: float,
) -> str:
    lambda0_str = "nan" if lambda0_energy is None else f"{lambda0_energy: .12f}"
    lambda1_str = "nan" if lambda1_energy is None else f"{lambda1_energy: .12f}"
    return (
        f"step={step:5d} time_fs={time_fs:9.3f} "
        f"temp={temperature:8.3f} K "
        f"energy={total_energy: .12f} eV "
        f"lambda0_energy={lambda0_str} eV "
        f"lambda1_energy={lambda1_str} eV "
        f"max|F|={max_force:.6f} "
        f"lambda={lambda_mask_to_string(lambda_mask)}"
    )


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

    with tee_output(log_path):
        model, ghost_model = load_md_models(args)
        print(
            f"loaded model: family={model.family}, name={model.model_name}, device={model.device}"
        )
        print(f"target indices: {target_indices}")
        print(f"target lambda: {args.lambda_value:.6f}")
        if should_enable_uma_merge_experts(args):
            print("UMA MOE merge: enabled")
            if ghost_model is not None:
                print("UMA MOE merge: using a second predictor for the ghost endpoint")
        elif args.model_type is not None and args.model_type.strip().lower() == "uma":
            print("UMA MOE merge: disabled")
        print(f"use D3 correction: {args.use_d3}")
        if args.use_d3:
            print(f"D3 method: {args.d3_method}, damping: {args.d3_damping}")

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
            use_d3=args.use_d3,
            d3_method=args.d3_method,
            d3_damping=args.d3_damping,
        )
        atoms.calc = calc

        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        print(f"initial corrected energy: {initial_energy:.12f} eV")
        print(f"initial lambda=0 energy: {calc.last_real_endpoint_energy:.12f} eV")
        print(f"initial lambda=1 energy: {calc.last_ghost_endpoint_energy:.12f} eV")
        print(f"initial max|F|: {np.abs(initial_forces).max():.6f} eV/A")
        print(
            f"initial lambda mask: {lambda_mask_to_string(np.asarray(atoms.arrays[args.lambda_array_name], dtype=np.float64))}"
        )

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

        def record_step() -> None:
            step = dynamics.nsteps
            time_fs = step * args.timestep_fs
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            temperature = atoms.get_temperature()
            lambda0_energy = calc.last_real_endpoint_energy
            lambda1_energy = calc.last_ghost_endpoint_energy
            lambda_mask_current = np.asarray(
                atoms.arrays[args.lambda_array_name], dtype=np.float64
            )
            max_force = float(np.abs(forces).max())

            frame = atoms.copy()
            annotate_frame_metadata(
                frame,
                step=step,
                time_fs=time_fs,
                temperature=temperature,
                total_energy=energy,
                lambda0_energy=lambda0_energy,
                lambda1_energy=lambda1_energy,
                lambda_array_name=args.lambda_array_name,
            )
            write(
                trajectory_path,
                frame,
                append=True,
                format="extxyz",
            )
            print(
                format_step_log(
                    step=step,
                    time_fs=time_fs,
                    temperature=temperature,
                    total_energy=energy,
                    lambda0_energy=lambda0_energy,
                    lambda1_energy=lambda1_energy,
                    lambda_mask=lambda_mask_current,
                    max_force=max_force,
                )
            )

        dynamics.attach(record_step, interval=args.log_interval)

        print(
            f"running ghost-target MD: steps={args.steps}, timestep_fs={args.timestep_fs}, "
            f"temperature_K={args.temperature}"
        )

        start_time = time.time()
        dynamics.run(args.steps)
        end_time = time.time()
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
            lambda_array_name=args.lambda_array_name,
        )
        write(final_structure_path, final_frame)
        print(f"trajectory written to {trajectory_path}")
        print(f"log written to {log_path}")
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
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--trajectory-name", default="ghost_md.xyz")
    parser.add_argument("--final-structure-name", default="ghost_final.extxyz")
    args = parser.parse_args()
    if args.ckpt_path is None and args.model_type is None:
        parser.error("Either --model-type or --ckpt-path must be provided.")
    return args


if __name__ == "__main__":
    main(parse_args())
