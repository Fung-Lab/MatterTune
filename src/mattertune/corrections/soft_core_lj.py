from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms
from numpy.typing import NDArray


def _resolve_target_mask(
    natoms: int,
    *,
    target_index: int | None,
    target_mask: Sequence[bool] | NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    if target_index is not None and target_mask is not None:
        raise ValueError("Specify only one of target_index or target_mask.")

    if target_index is not None:
        if target_index < 0 or target_index >= natoms:
            raise ValueError(
                f"target_index must lie in [0, {natoms - 1}], got {target_index}."
            )
        mask = np.zeros(natoms, dtype=bool)
        mask[target_index] = True
        return mask

    if target_mask is None:
        return np.zeros(natoms, dtype=bool)

    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != (natoms,):
        raise ValueError(
            f"Expected target_mask with shape ({natoms},), got {mask.shape}."
        )
    return mask


def _lj_pair_energy(
    r2: NDArray[np.float64],
    *,
    epsilon: float,
    sigma: float,
) -> NDArray[np.float64]:
    reduced_r6 = (r2 / sigma**2) ** 3
    return 4.0 * epsilon * (1.0 / reduced_r6**2 - 1.0 / reduced_r6)


def _lj_pair_force_scalar(
    r2: NDArray[np.float64],
    *,
    epsilon: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Return the ASE-style scalar prefactor so that F_ij = scalar * d_ij."""
    reduced_r6 = (r2 / sigma**2) ** 3
    return 24.0 * epsilon * (-2.0 / reduced_r6**2 + 1.0 / reduced_r6) / r2


def soft_core_lj_correction(
    atoms: Atoms,
    *,
    target_index: int | None = None,
    target_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
    epsilon: float = 1.0,
    sigma: float = 1.0,
    alpha: float = 0.5,
    rc: float | None = None,
    ro: float | None = None,
    smooth: bool = False,
) -> tuple[float, NDArray[np.float64]]:
    """Compute a target-specific Lennard-Jones correction.

    This correction is designed for ghost-like reference states: it acts only
    on pairs containing one explicitly selected target atom and one non-target
    environment atom. Environment-environment pairs and target-target pairs
    are left unchanged.

    The pair potential uses the classical 12-6 Lennard-Jones form,

    ``V_LJ(r) = 4 * epsilon * ((sigma / r)^12 - (sigma / r)^6)``.

    In the electrolyte example workflow, this correction is attached only to
    the fully ghost endpoint. Intermediate lambda values do not evaluate a
    separate lambda-dependent LJ term; they inherit this contribution only
    through endpoint interpolation.

    No cutoff, energy shift, smoothing, or soft-core outer transform is applied.
    ``alpha``, ``rc``, ``ro``, and ``smooth`` are retained in the public
    signature for backward-compatible example scripts, but they are ignored by
    this pure 12-6 LJ implementation.
    """
    _ = (alpha, rc, ro, smooth)

    natoms = len(atoms)
    resolved_target_mask = _resolve_target_mask(
        natoms,
        target_index=target_index,
        target_mask=target_mask,
    )
    if not np.any(resolved_target_mask):
        return 0.0, np.zeros((natoms, 3), dtype=np.float64)

    forces = np.zeros((natoms, 3), dtype=np.float64)
    energy = 0.0

    target_indices = np.flatnonzero(resolved_target_mask)
    environment_indices = np.flatnonzero(~resolved_target_mask)
    if len(environment_indices) == 0:
        return 0.0, forces

    for ii in target_indices:
        distance_vectors = atoms.get_distances(
            int(ii),
            environment_indices,
            mic=True,
            vector=True,
        )
        r2 = np.einsum("ij,ij->i", distance_vectors, distance_vectors)
        if np.any(r2 <= 0.0):
            raise ValueError(
                "LJ correction encountered a zero-distance target-environment pair."
            )

        pairwise_energies = _lj_pair_energy(
            r2,
            epsilon=epsilon,
            sigma=sigma,
        )
        pairwise_forces = _lj_pair_force_scalar(
            r2,
            epsilon=epsilon,
            sigma=sigma,
        )

        pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors

        energy += pairwise_energies.sum()
        forces[ii] += pairwise_forces.sum(axis=0)
        forces[environment_indices] -= pairwise_forces

    return float(energy), forces


def _assert_allclose(
    lhs: NDArray[np.float64] | float,
    rhs: NDArray[np.float64] | float,
    *,
    atol: float,
    message: str,
) -> None:
    if not np.allclose(lhs, rhs, atol=atol, rtol=0.0):
        raise SystemExit(f"{message}\nleft={lhs}\nright={rhs}")


if __name__ == "__main__":
    sigma = 1.0
    epsilon = 1.0
    alpha = 0.5
    rc = 3.0
    ro = 2.0

    triad = Atoms(
        "Ar3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )

    zero_energy, zero_forces = soft_core_lj_correction(
        triad,
        epsilon=epsilon,
        sigma=sigma,
        alpha=alpha,
        rc=rc,
        ro=ro,
        smooth=False,
    )
    _assert_allclose(
        zero_energy,
        0.0,
        atol=0.0,
        message="No selected target atoms should give zero correction energy.",
    )
    _assert_allclose(
        zero_forces,
        np.zeros((3, 3), dtype=np.float64),
        atol=0.0,
        message="No selected target atoms should give zero correction forces.",
    )

    target_only_energy, target_only_forces = soft_core_lj_correction(
        triad,
        target_index=0,
        epsilon=epsilon,
        sigma=sigma,
        alpha=alpha,
        rc=rc,
        ro=ro,
        smooth=False,
    )
    target_distances2 = np.array([1.1**2, 2.0**2], dtype=np.float64)
    expected_pair_energies = _lj_pair_energy(
        target_distances2,
        epsilon=epsilon,
        sigma=sigma,
    )
    _assert_allclose(
        target_only_energy,
        float(expected_pair_energies.sum()),
        atol=1e-12,
        message="Only target-environment pairs should contribute to the correction energy.",
    )

    overlap_atoms = Atoms(
        "Ar2",
        positions=[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    overlap_energy, overlap_forces = soft_core_lj_correction(
        overlap_atoms,
        target_index=0,
        epsilon=epsilon,
        sigma=sigma,
        alpha=alpha,
        rc=rc,
        ro=ro,
        smooth=False,
    )
    if overlap_energy <= 0.0:
        raise SystemExit("Short-range target-environment overlap should be repulsive.")
    if overlap_forces[0, 0] >= 0.0 or overlap_forces[1, 0] <= 0.0:
        raise SystemExit("Short-range target-environment forces should push the atoms apart.")

    all_targets_energy, all_targets_forces = soft_core_lj_correction(
        overlap_atoms,
        target_mask=[True, True],
        epsilon=epsilon,
        sigma=sigma,
        alpha=alpha,
        rc=rc,
        ro=ro,
        smooth=False,
    )
    _assert_allclose(
        all_targets_energy,
        0.0,
        atol=0.0,
        message="Target-target pairs should not receive the correction.",
    )
    _assert_allclose(
        all_targets_forces,
        np.zeros((2, 3), dtype=np.float64),
        atol=0.0,
        message="Selecting every atom as a target should leave no target-environment pairs.",
    )

    ignored_options_energy, ignored_options_forces = soft_core_lj_correction(
        overlap_atoms,
        target_index=0,
        epsilon=epsilon,
        sigma=sigma,
        alpha=alpha,
        rc=rc,
        ro=ro,
        smooth=True,
    )
    _assert_allclose(
        ignored_options_energy,
        overlap_energy,
        atol=1e-12,
        message="alpha/rc/ro/smooth should not change the pure LJ correction energy.",
    )
    _assert_allclose(
        ignored_options_forces,
        overlap_forces,
        atol=1e-12,
        message="alpha/rc/ro/smooth should not change the pure LJ correction forces.",
    )

    print("All soft-core target correction checks passed.")
