from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.calculators.lj import cutoff_function
from ase.calculators.lj import d_cutoff_function
from ase.neighborlist import NeighborList
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


def _repulsive_soft_core_pair_energy(
    r2: NDArray[np.float64],
    *,
    epsilon: float,
    sigma: float,
) -> NDArray[np.float64]:
    reduced_r6 = (r2 / sigma**2) ** 3
    return 4.0 * epsilon * (1.0 / reduced_r6**2 - 1.0 / reduced_r6)


def _repulsive_soft_core_pair_force_scalar(
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
    """Compute a target-specific repulsive soft-core LJ correction.

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

    ``alpha`` is retained in the public signature for backward compatibility
    with existing example scripts, but it is not used in this ghost-only form.

    The implementation follows ASE ``LennardJones`` conventions for
    neighbor-list construction, pair-energy partitioning, and cutoff handling.
    When ``smooth=False``, the pair energy is shifted by its value at ``rc``.
    When ``smooth=True``, the same ASE smooth cutoff function is applied to the
    corrected target-environment potential.
    """
    natoms = len(atoms)
    resolved_target_mask = _resolve_target_mask(
        natoms,
        target_index=target_index,
        target_mask=target_mask,
    )
    if not np.any(resolved_target_mask):
        return 0.0, np.zeros((natoms, 3), dtype=np.float64)

    if rc is None:
        rc = 3.0 * sigma
    if ro is None:
        ro = 0.66 * rc

    nl = NeighborList([rc / 2.0] * natoms, self_interaction=False, bothways=True)
    nl.update(atoms)

    positions = atoms.positions
    cell = atoms.cell

    forces = np.zeros((natoms, 3), dtype=np.float64)
    energy = 0.0
    rc2 = rc**2

    for ii in range(natoms):
        neighbors, offsets = nl.get_neighbors(ii)
        if len(neighbors) == 0:
            continue

        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[ii]
        r2 = np.einsum("ij,ij->i", distance_vectors, distance_vectors)

        ii_is_target = resolved_target_mask[ii]
        neighbor_is_target = resolved_target_mask[neighbors]
        target_environment_mask = (ii_is_target != neighbor_is_target) & (r2 <= rc2)
        if not np.any(target_environment_mask):
            continue

        distance_vectors = distance_vectors[target_environment_mask]
        r2 = r2[target_environment_mask]

        pairwise_energies = _repulsive_soft_core_pair_energy(
            r2,
            epsilon=epsilon,
            sigma=sigma,
        )
        pairwise_forces = _repulsive_soft_core_pair_force_scalar(
            r2,
            epsilon=epsilon,
            sigma=sigma,
        )

        if smooth:
            cutoff_fn = cutoff_function(r2, rc2, ro**2)
            d_cutoff_fn = d_cutoff_function(r2, rc2, ro**2)
            pairwise_forces = (
                cutoff_fn * pairwise_forces + 2.0 * d_cutoff_fn * pairwise_energies
            )
            pairwise_energies *= cutoff_fn
        else:
            pairwise_energies -= _repulsive_soft_core_pair_energy(
                np.full_like(r2, rc2, dtype=np.float64),
                epsilon=epsilon,
                sigma=sigma,
            )

        pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors

        energy += 0.5 * pairwise_energies.sum()
        forces[ii] += pairwise_forces.sum(axis=0)

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
    from ase.calculators.lj import LennardJones

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
    expected_pair_energies = _repulsive_soft_core_pair_energy(
        target_distances2,
        epsilon=epsilon,
        sigma=sigma,
    ) - _repulsive_soft_core_pair_energy(
        np.full(2, rc**2, dtype=np.float64),
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
    reference_atoms = overlap_atoms.copy()
    reference_atoms.calc = LennardJones(
        epsilon=epsilon,
        sigma=sigma,
        rc=rc,
        ro=ro,
        smooth=False,
    )
    reference_lj_energy = reference_atoms.get_potential_energy()
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
    if np.isclose(all_targets_energy, reference_lj_energy, atol=1e-12, rtol=0.0):
        raise SystemExit(
            "The ghost-state correction must not reproduce the full-system ASE Lennard-Jones energy."
        )

    smooth_energy, smooth_forces = soft_core_lj_correction(
        overlap_atoms,
        target_index=0,
        epsilon=epsilon,
        sigma=sigma,
        alpha=alpha,
        rc=rc,
        ro=ro,
        smooth=True,
    )
    if not np.isfinite(smooth_energy) or not np.all(np.isfinite(smooth_forces)):
        raise SystemExit("Smooth cutoff mode should produce finite energy and forces.")

    print("All soft-core target correction checks passed.")
