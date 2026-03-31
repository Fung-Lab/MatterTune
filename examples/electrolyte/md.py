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
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from mattertune import load_pretrained_model

from ghost_target_calculator import GhostTargetCorrectionCalculator


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


def main(args: argparse.Namespace) -> None:
    atoms = load_atoms(args.structure)
    target_indices = parse_target_indices(args.target_indices, len(atoms))
    target_mask = build_target_mask(len(atoms), target_indices)
    lambda_mask = build_lambda_mask(len(atoms), target_indices, args.lambda_value)
    atoms.arrays[args.lambda_array_name] = lambda_mask
    atoms.arrays[args.target_array_name] = target_mask

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
        f"loaded model: family={model.family}, name={model.model_name}, device={model.device}"
    )
    print(f"target indices: {target_indices}")
    print(f"target lambda: {args.lambda_value:.6f}")

    calc = GhostTargetCorrectionCalculator(
        model,
        lambda_array_name=args.lambda_array_name,
        target_array_name=args.target_array_name,
        epsilon=args.epsilon,
        sigma=args.sigma,
        alpha=args.alpha,
        rc=args.rc,
        ro=args.ro,
        smooth=args.smooth,
    )
    atoms.calc = calc

    initial_energy = atoms.get_potential_energy()
    initial_forces = atoms.get_forces()
    print(f"initial corrected energy: {initial_energy:.12f} eV")
    print(f"initial max|F|: {np.abs(initial_forces).max():.6f} eV/A")

    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=args.temperature,
        rng=np.random.default_rng(args.seed),
    )
    Stationary(atoms)

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
        max_force = np.abs(atoms.get_forces()).max()
        print(
            f"step={step:5d} time_fs={step * args.timestep_fs:9.3f} "
            f"energy={energy: .12f} eV temp={temperature:8.3f} K max|F|={max_force:.6f}"
        )

    dynamics.attach(trajectory.write, interval=args.log_interval)
    dynamics.attach(log_step, interval=args.log_interval)

    print(
        f"running ghost-target MD: steps={args.steps}, timestep_fs={args.timestep_fs}, "
        f"temperature_K={args.temperature}"
    )
    dynamics.run(args.steps)
    trajectory.close()

    write(final_structure_path, atoms)
    print(f"trajectory written to {trajectory_path}")
    print(f"final structure written to {final_structure_path}")

    traj = read(trajectory_path, index=":")
    write(trajectory_path, traj)


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parent / "outputs" / "md"
    parser = argparse.ArgumentParser(
        description="Run MD with a pretrained model plus ghost-target excluded-volume correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
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
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--friction-fs-inv", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--trajectory-name", default="ghost_md.xyz")
    parser.add_argument("--final-structure-name", default="ghost_final.extxyz")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
