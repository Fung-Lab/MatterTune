from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms

from mattertune.corrections import soft_core_lj_correction


def evaluate_pair_energy_and_force(
    distance: float,
    *,
    epsilon: float,
    sigma: float,
    alpha: float,
    rc: float,
    ro: float,
) -> tuple[float, float]:
    atoms = Atoms(
        "Ar2",
        positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    energy, forces = soft_core_lj_correction(
        atoms,
        target_index=0,
        epsilon=epsilon,
        sigma=sigma,
        alpha=alpha,
        rc=rc,
        ro=ro,
        smooth=True,
    )
    return float(energy), float(forces[1, 0])


def finite_difference_force(
    distance: float,
    *,
    step: float,
    epsilon: float,
    sigma: float,
    alpha: float,
    rc: float,
    ro: float,
) -> float:
    energy_plus, _ = evaluate_pair_energy_and_force(
        distance + step,
        epsilon=epsilon,
        sigma=sigma,
        alpha=alpha,
        rc=rc,
        ro=ro,
    )
    energy_minus, _ = evaluate_pair_energy_and_force(
        distance - step,
        epsilon=epsilon,
        sigma=sigma,
        alpha=alpha,
        rc=rc,
        ro=ro,
    )
    return -(energy_plus - energy_minus) / (2.0 * step)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_distances = np.array(
        [
            args.rc - 1.0e-2,
            args.rc - 1.0e-3,
            args.rc - 1.0e-4,
            args.rc,
            args.rc + 1.0e-4,
            args.rc + 1.0e-3,
            args.rc + 1.0e-2,
        ],
        dtype=np.float64,
    )

    continuity_rows: list[tuple[float, float, float]] = []
    print("Pure-LJ scan around the diagnostic rc value (rc is ignored by the correction)")
    for distance in sample_distances:
        energy, force_x = evaluate_pair_energy_and_force(
            distance,
            epsilon=args.epsilon,
            sigma=args.sigma,
            alpha=args.alpha,
            rc=args.rc,
            ro=args.ro,
        )
        continuity_rows.append((distance, energy, force_x))
        print(
            f"r={distance:.8f} A  energy={energy: .12e} eV  force_x={force_x: .12e} eV/A"
        )

    left_energy = continuity_rows[2][1]
    at_energy = continuity_rows[3][1]
    right_energy = continuity_rows[4][1]
    left_force = continuity_rows[2][2]
    at_force = continuity_rows[3][2]
    right_force = continuity_rows[4][2]

    max_cutoff_energy_jump = max(abs(left_energy - at_energy), abs(right_energy - at_energy))
    max_cutoff_force_jump = max(abs(left_force - at_force), abs(right_force - at_force))
    print(f"max energy change near diagnostic rc: {max_cutoff_energy_jump:.12e} eV")
    print(f"max force change near diagnostic rc: {max_cutoff_force_jump:.12e} eV/A")

    gradient_rows: list[tuple[float, float, float, float]] = []
    print("\nFinite-difference force/gradient check")
    for distance in np.linspace(args.ro + 0.1, args.rc - 0.05, args.num_gradient_points):
        energy, force_x = evaluate_pair_energy_and_force(
            distance,
            epsilon=args.epsilon,
            sigma=args.sigma,
            alpha=args.alpha,
            rc=args.rc,
            ro=args.ro,
        )
        fd_force_x = finite_difference_force(
            distance,
            step=args.fd_step,
            epsilon=args.epsilon,
            sigma=args.sigma,
            alpha=args.alpha,
            rc=args.rc,
            ro=args.ro,
        )
        abs_diff = abs(force_x - fd_force_x)
        gradient_rows.append((distance, energy, force_x, fd_force_x))
        print(
            f"r={distance:.8f} A  analytic={force_x: .12e}  "
            f"fd={fd_force_x: .12e}  abs_diff={abs_diff:.12e}"
        )

    max_gradient_error = max(abs(force - fd) for _, _, force, fd in gradient_rows)
    print(f"max |F + dE/dr| error: {max_gradient_error:.12e} eV/A")
    if max_gradient_error > args.gradient_tolerance:
        raise SystemExit(
            f"Finite-difference force check failed: {max_gradient_error} > {args.gradient_tolerance}"
        )

    continuity_path = output_dir / "pure_lj_scan.csv"
    with continuity_path.open("w", encoding="utf-8") as handle:
        handle.write("distance,energy,force_x\n")
        for distance, energy, force_x in continuity_rows:
            handle.write(f"{distance:.16f},{energy:.16e},{force_x:.16e}\n")

    gradient_path = output_dir / "gradient_check.csv"
    with gradient_path.open("w", encoding="utf-8") as handle:
        handle.write("distance,energy,analytic_force_x,fd_force_x\n")
        for distance, energy, force_x, fd_force_x in gradient_rows:
            handle.write(
                f"{distance:.16f},{energy:.16e},{force_x:.16e},{fd_force_x:.16e}\n"
            )

    print(f"\nWrote {continuity_path}")
    print(f"Wrote {gradient_path}")


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parent / "outputs" / "continuity"
    parser = argparse.ArgumentParser(
        description="Check pure LJ smoothness and force/energy consistency for the target correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rc", type=float, default=3.0)
    parser.add_argument("--ro", type=float, default=2.0)
    parser.add_argument("--fd-step", type=float, default=1.0e-5)
    parser.add_argument("--num-gradient-points", type=int, default=6)
    parser.add_argument("--gradient-tolerance", type=float, default=2.0e-6)
    parser.add_argument("--output-dir", default=str(default_output_dir))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
