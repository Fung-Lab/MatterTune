from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mattertune.pretrained import PretrainedModel
from mattertune.corrections import soft_core_lj_correction


def _normalize_binary_lambda_mask(
    lambda_mask: Sequence[float] | np.ndarray,
    natoms: int,
) -> np.ndarray:
    lambda_array = np.asarray(lambda_mask, dtype=np.float64)
    if lambda_array.shape != (natoms,):
        raise ValueError(
            f"Expected lambda mask with shape ({natoms},), got {lambda_array.shape}."
        )

    rounded = np.rint(lambda_array)
    if not np.allclose(lambda_array, rounded, atol=1e-8, rtol=0.0):
        raise ValueError(
            "This example calculator only supports binary lambda masks with values in {0, 1}."
        )
    if np.any((rounded != 0.0) & (rounded != 1.0)):
        raise ValueError(
            "This example calculator only supports binary lambda masks with values in {0, 1}."
        )
    return rounded.astype(np.float64)


def _copy_atoms(atoms: Atoms) -> Atoms:
    atoms_copy = atoms.copy()
    atoms_copy.info = copy.deepcopy(atoms.info)
    return atoms_copy


def _slice_atoms(atoms: Atoms, keep_mask: np.ndarray) -> Atoms:
    reduced = atoms[np.flatnonzero(keep_mask)].copy()
    reduced.info = copy.deepcopy(atoms.info)
    return reduced


def _scalar_energy(value: object) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


class GhostTargetCorrectionCalculator(Calculator):
    """Delete lambda=0 target atoms from the model input, then add a repulsive correction."""

    def __init__(
        self,
        model: PretrainedModel,
        *,
        lambda_mask: Sequence[float] | np.ndarray | None = None,
        lambda_array_name: str = "alchemical_lambda",
        epsilon: float = 1.0,
        sigma: float = 1.0,
        alpha: float = 0.5,
        rc: float | None = None,
        ro: float | None = None,
        smooth: bool = True,
    ):
        super().__init__()
        if "energy" not in model.implemented_properties:
            raise ValueError("The wrapped pretrained model must implement `energy`.")
        if "forces" not in model.implemented_properties:
            raise ValueError("The wrapped pretrained model must implement `forces`.")

        self._model = model
        self._fixed_lambda_mask = None if lambda_mask is None else np.asarray(
            lambda_mask, dtype=np.float64)
        self._lambda_array_name = lambda_array_name
        self._correction_kwargs = {
            "epsilon": epsilon,
            "sigma": sigma,
            "alpha": alpha,
            "rc": rc,
            "ro": ro,
            "smooth": smooth,
        }
        self.implemented_properties = ["energy", "forces", "free_energy"]

    def _resolve_lambda_mask(self, atoms: Atoms) -> np.ndarray:
        if self._fixed_lambda_mask is not None:
            return _normalize_binary_lambda_mask(self._fixed_lambda_mask, len(atoms))

        if self._lambda_array_name not in atoms.arrays:
            raise ValueError(
                "No lambda mask was provided to the calculator and "
                f"`atoms.arrays[{self._lambda_array_name!r}]` is missing."
            )
        return _normalize_binary_lambda_mask(atoms.arrays[self._lambda_array_name], len(atoms))

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        requested = properties or list(self.implemented_properties)
        unsupported = sorted(set(requested) - set(self.implemented_properties))
        if unsupported:
            raise ValueError(
                f"Unsupported properties requested: {unsupported}. "
                f"Supported properties are {self.implemented_properties}."
            )

        Calculator.calculate(self, atoms, requested, system_changes or all_changes)
        assert isinstance(self.atoms, Atoms)

        full_atoms = _copy_atoms(self.atoms)
        lambda_mask = self._resolve_lambda_mask(full_atoms)
        ghost_mask = np.isclose(lambda_mask, 0.0)
        keep_mask = ~ghost_mask

        natoms = len(full_atoms)
        full_forces = np.zeros((natoms, 3), dtype=np.float64)
        base_energy = 0.0

        if np.any(keep_mask):
            reduced_atoms = _slice_atoms(full_atoms, keep_mask)
            base_prediction = self._model.predict_one(
                reduced_atoms,
                properties=["energy", "forces"],
            )
            base_energy = _scalar_energy(base_prediction["energy"])
            full_forces[keep_mask] = np.asarray(
                base_prediction["forces"],
                dtype=np.float64,
            )

        correction_energy, correction_forces = soft_core_lj_correction(
            full_atoms,
            lambda_mask,
            target_mask=ghost_mask,
            **self._correction_kwargs,
        )

        total_energy = base_energy + correction_energy
        total_forces = full_forces + correction_forces

        if "energy" in requested or "free_energy" in requested:
            self.results["energy"] = float(total_energy)
            self.results["free_energy"] = float(total_energy)
        if "forces" in requested:
            self.results["forces"] = total_forces


__all__ = ["GhostTargetCorrectionCalculator"]
