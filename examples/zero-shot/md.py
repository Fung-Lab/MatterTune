from __future__ import annotations

import argparse
from pathlib import Path

import ase.units as units
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from mattertune import available_pretrained_models, load_pretrained_model


def build_default_atoms() -> Atoms:
    atoms = bulk("Si", "diamond", a=5.43)
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


def describe_prediction(prediction: dict[str, object]) -> None:
    print(f"prediction keys: {sorted(prediction)}")
    if "energy" in prediction:
        energy = float(np.asarray(prediction["energy"]).reshape(-1)[0])
        print(f"predicted energy: {energy:.12f} eV")
    if "forces" in prediction:
        forces = np.asarray(prediction["forces"])
        print(
            "predicted forces: "
            f"shape={forces.shape}, max|F|={np.abs(forces).max():.6f} eV/Ang"
        )


def main(args: argparse.Namespace) -> None:
    if args.list_models:
        print(available_pretrained_models(args.model_type))
        return

    atoms = load_atoms(args.structure)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, object] = {}
    if args.task_name is not None:
        load_kwargs["task_name"] = args.task_name

    model = load_pretrained_model(
        args.model_type,
        args.model_name,
        device=args.device,
        **load_kwargs,
    )
    print(
        f"loaded model: family={model.family}, name={model.model_name}, device={model.device}")
    print(f"implemented properties: {model.implemented_properties}")

    prediction = model.predict(atoms)
    assert isinstance(prediction, dict)
    describe_prediction(prediction)

    calc = model.ase_calculator()
    atoms.calc = calc

    MaxwellBoltzmannDistribution(
        atoms, temperature_K=args.temperature, rng=np.random.default_rng(args.seed))
    Stationary(atoms)
    if not np.any(atoms.pbc):
        ZeroRotation(atoms)

    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=args.friction_fs_inv / units.fs,
        fixcm=False,
    )

    trajectory_path = output_dir / args.trajectory_name
    final_structure_path = output_dir / args.final_structure_name
    trajectory = Trajectory(str(trajectory_path), "w", atoms)

    def log_step() -> None:
        step = dynamics.nsteps
        energy = atoms.get_potential_energy()
        temperature = atoms.get_temperature()
        print(
            f"step={step:5d} time_fs={step * args.timestep_fs:9.3f} "
            f"energy={energy: .12f} eV temp={temperature:8.3f} K"
        )

    dynamics.attach(trajectory.write, interval=args.log_interval)
    dynamics.attach(log_step, interval=args.log_interval)

    print(
        f"running Langevin MD: steps={args.steps}, "
        f"timestep_fs={args.timestep_fs}, temperature_K={args.temperature}"
    )
    dynamics.run(args.steps)
    trajectory.close()

    write(final_structure_path, atoms)
    print(f"trajectory written to {trajectory_path}")
    print(f"final structure written to {final_structure_path}")


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parent / "outputs"

    parser = argparse.ArgumentParser(
        description="Run zero-shot MD with a MatterTune pretrained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-type", required=True,
                        help="mattersim, orb, mace, nequip, allegro, uma")
    parser.add_argument("--model-name", default=None,
                        help="Optional pretrained model name. Uses MatterTune defaults when omitted.")
    parser.add_argument("--task-name", default=None,
                        help="Required for UMA, e.g. omat or omol.")
    parser.add_argument("--device", default="cpu",
                        help="Inference device, e.g. cpu or cuda:0.")
    parser.add_argument("--structure", default=None,
                        help="Optional input structure readable by ASE. Defaults to bulk Si.")
    parser.add_argument("--temperature", type=float,
                        default=300.0, help="Target temperature in K.")
    parser.add_argument("--timestep-fs", type=float,
                        default=1.0, help="MD timestep in fs.")
    parser.add_argument("--friction-fs-inv", type=float,
                        default=0.02, help="Langevin friction in 1/fs.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of MD steps.")
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Logging and trajectory write interval.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for velocity initialization.")
    parser.add_argument("--output-dir", default=str(default_output_dir),
                        help="Directory for trajectory and final structure outputs.")
    parser.add_argument("--trajectory-name", default="md.traj",
                        help="Output ASE trajectory filename.")
    parser.add_argument("--final-structure-name", default="final.extxyz",
                        help="Output filename for the final structure.")
    parser.add_argument("--list-models", action="store_true",
                        help="List available pretrained models for the selected family and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
