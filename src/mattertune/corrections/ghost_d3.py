from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from ..alchemical import EnergyForcePrediction
from ..alchemical import interpolate_energy_forces
from ..alchemical import validate_lambda_value


def _copy_atoms(atoms: Atoms) -> Atoms:
    atoms_copy = atoms.copy()
    atoms_copy.info = copy.deepcopy(atoms.info)
    return atoms_copy


def _slice_atoms(atoms: Atoms, keep_mask: NDArray[np.bool_]) -> Atoms:
    reduced = atoms[np.flatnonzero(keep_mask)].copy()
    reduced.info = copy.deepcopy(atoms.info)
    return reduced


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


def _run_d3(
    atoms: Atoms,
    *,
    method: str,
    damping: str,
) -> EnergyForcePrediction:
    try:
        from dftd3.ase import DFTD3
    except ImportError as exc:
        raise ImportError(
            "ghost_target_d3_correction requires the `dftd3` Python package. "
            "Install it first, for example with `pip install dftd3`."
        ) from exc

    atoms_copy = _copy_atoms(atoms)
    atoms_copy.calc = DFTD3(method=method, damping=damping)
    return EnergyForcePrediction(
        energy=float(atoms_copy.get_potential_energy()),
        forces=np.asarray(atoms_copy.get_forces(), dtype=np.float64),
    )


def ghost_target_d3_correction(
    atoms: Atoms,
    target_lambda: float,
    *,
    target_index: int | None = None,
    target_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
    method: str = "pbe",
    damping: str = "d3bj",
) -> tuple[float, NDArray[np.float64]]:
    """Compute a ghost-aware D3 correction consistent with endpoint deletion.

    Endpoint semantics:
    - lambda = 0: evaluate D3 on the full system
    - lambda = 1: evaluate D3 on the reduced system with target atoms removed,
      then pad target forces with zeros
    - 0 < lambda < 1: linearly interpolate between those two endpoint results
    """
    lam = validate_lambda_value(target_lambda)
    natoms = len(atoms)
    resolved_target_mask = _resolve_target_mask(
        natoms,
        target_index=target_index,
        target_mask=target_mask,
    )

    real_prediction = _run_d3(
        atoms,
        method=method,
        damping=damping,
    )

    if not np.any(resolved_target_mask) or lam <= 1e-8:
        return real_prediction.energy, real_prediction.forces

    keep_mask = ~resolved_target_mask
    ghost_forces = np.zeros((natoms, 3), dtype=np.float64)
    ghost_energy = 0.0
    if np.any(keep_mask):
        reduced_prediction = _run_d3(
            _slice_atoms(atoms, keep_mask),
            method=method,
            damping=damping,
        )
        ghost_energy = reduced_prediction.energy
        ghost_forces[keep_mask] = reduced_prediction.forces
    ghost_prediction = EnergyForcePrediction(
        energy=ghost_energy,
        forces=ghost_forces,
    )

    if lam >= 1.0 - 1e-8:
        return ghost_prediction.energy, ghost_prediction.forces

    interpolated = interpolate_energy_forces(
        real_prediction,
        ghost_prediction,
        lam,
    )
    return interpolated.energy, interpolated.forces
