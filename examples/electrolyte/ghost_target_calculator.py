from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mattertune.alchemical import EnergyForcePrediction
from mattertune.alchemical import interpolate_energy_forces
from mattertune.pretrained import PretrainedModel
from mattertune.corrections import ghost_target_d3_correction
from mattertune.corrections import soft_core_lj_correction


def _normalize_lambda_mask(
    lambda_mask: Sequence[float] | np.ndarray,
    natoms: int,
) -> np.ndarray:
    lambda_array = np.asarray(lambda_mask, dtype=np.float64)
    if lambda_array.shape != (natoms,):
        raise ValueError(
            f"Expected lambda mask with shape ({natoms},), got {lambda_array.shape}."
        )
    if np.any(lambda_array < 0.0) or np.any(lambda_array > 1.0):
        raise ValueError(
            "This example calculator only supports lambda masks with values in [0, 1]."
        )
    return lambda_array


def _normalize_target_mask(
    target_mask: Sequence[bool] | np.ndarray,
    natoms: int,
) -> np.ndarray:
    target_array = np.asarray(target_mask, dtype=bool)
    if target_array.shape != (natoms,):
        raise ValueError(
            f"Expected target mask with shape ({natoms},), got {target_array.shape}."
        )
    return target_array


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


def _force_norms(forces: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    return {
        "target_force_norm_eVA": float(np.linalg.norm(forces[target_mask])),
        "environment_force_norm_eVA": float(np.linalg.norm(forces[~target_mask])),
        "max_force_eVA": float(np.max(np.abs(forces))) if forces.size else 0.0,
    }


class GhostTargetCorrectionCalculator(Calculator):
    """Interpolate between real-target and fully-ghost target endpoints.

    Calculator-level lambda semantics:
    - lambda = 0: the selected target atoms are fully real and remain in the model input
    - lambda = 1: the selected target atoms are evaluated on the configured ghost endpoint
    - 0 < lambda < 1: interpolate linearly between those two endpoint predictions
    """

    def __init__(
        self,
        model: PretrainedModel,
        *,
        ghost_model: PretrainedModel | None = None,
        lambda_mask: Sequence[float] | np.ndarray | None = None,
        target_mask: Sequence[bool] | np.ndarray | None = None,
        lambda_array_name: str = "alchemical_lambda",
        target_array_name: str = "alchemical_target",
        epsilon: float = 1.0,
        sigma: float = 1.0,
        alpha: float = 0.5,
        rc: float | None = None,
        ro: float | None = None,
        smooth: bool = True,
        ghost_endpoint_mode: str = "delete",
        use_d3: bool = False,
        d3_method: str = "pbe",
        d3_damping: str = "d3bj",
    ):
        super().__init__()
        if "energy" not in model.implemented_properties:
            raise ValueError("The wrapped pretrained model must implement `energy`.")
        if "forces" not in model.implemented_properties:
            raise ValueError("The wrapped pretrained model must implement `forces`.")
        if ghost_model is not None:
            if "energy" not in ghost_model.implemented_properties:
                raise ValueError(
                    "The ghost-endpoint pretrained model must implement `energy`."
                )
            if "forces" not in ghost_model.implemented_properties:
                raise ValueError(
                    "The ghost-endpoint pretrained model must implement `forces`."
                )

        self._model = model
        self._ghost_model = ghost_model
        self._fixed_lambda_mask = None if lambda_mask is None else np.asarray(
            lambda_mask, dtype=np.float64)
        self._fixed_target_mask = None if target_mask is None else np.asarray(
            target_mask, dtype=bool)
        self._lambda_array_name = lambda_array_name
        self._target_array_name = target_array_name
        self._correction_kwargs = {
            "epsilon": epsilon,
            "sigma": sigma,
            "alpha": alpha,
            "rc": rc,
            "ro": ro,
            "smooth": smooth,
        }
        if ghost_endpoint_mode not in {"delete", "dummy"}:
            raise ValueError(
                "ghost_endpoint_mode must be either `delete` or `dummy`."
            )
        self._ghost_endpoint_mode = ghost_endpoint_mode
        self._use_d3 = use_d3
        self._d3_kwargs = {
            "method": d3_method,
            "damping": d3_damping,
        }
        self._last_real_endpoint_energy: float | None = None
        self._last_ghost_endpoint_energy: float | None = None
        self._last_real_endpoint_details: dict[str, object] | None = None
        self._last_ghost_endpoint_details: dict[str, object] | None = None
        self.implemented_properties = ["energy", "forces", "free_energy"]

    @property
    def last_real_endpoint_energy(self) -> float | None:
        return self._last_real_endpoint_energy

    @property
    def last_ghost_endpoint_energy(self) -> float | None:
        return self._last_ghost_endpoint_energy

    @property
    def last_real_endpoint_details(self) -> dict[str, object] | None:
        return copy.deepcopy(self._last_real_endpoint_details)

    @property
    def last_ghost_endpoint_details(self) -> dict[str, object] | None:
        return copy.deepcopy(self._last_ghost_endpoint_details)

    @property
    def last_ghost_endpoint_lj_energy(self) -> float | None:
        if self._last_ghost_endpoint_details is None:
            return None
        value = self._last_ghost_endpoint_details.get("lj_energy_eV")
        return None if value is None else float(value)

    def check_state(self, atoms, tol=1e-15):
        system_changes = super().check_state(atoms, tol=tol)
        if self.atoms is None:
            return system_changes

        for array_name in (self._lambda_array_name, self._target_array_name):
            previous = self.atoms.arrays.get(array_name)
            current = atoms.arrays.get(array_name)
            if previous is None and current is None:
                continue
            if previous is None or current is None:
                system_changes.append(array_name)
                continue
            if previous.shape != current.shape or not np.array_equal(previous, current):
                system_changes.append(array_name)
        return system_changes

    def _resolve_lambda_mask(self, atoms: Atoms) -> np.ndarray:
        if self._fixed_lambda_mask is not None:
            return _normalize_lambda_mask(self._fixed_lambda_mask, len(atoms))

        if self._lambda_array_name not in atoms.arrays:
            raise ValueError(
                "No lambda mask was provided to the calculator and "
                f"`atoms.arrays[{self._lambda_array_name!r}]` is missing."
            )
        return _normalize_lambda_mask(atoms.arrays[self._lambda_array_name], len(atoms))

    def _resolve_target_mask(self, atoms: Atoms, lambda_mask: np.ndarray) -> np.ndarray:
        if self._fixed_target_mask is not None:
            return _normalize_target_mask(self._fixed_target_mask, len(atoms))

        if self._target_array_name in atoms.arrays:
            return _normalize_target_mask(atoms.arrays[self._target_array_name], len(atoms))

        # Fallback: infer targets from strictly positive lambda values.
        # This is insufficient when target lambda is exactly zero, so callers should
        # provide an explicit target mask in that case.
        return lambda_mask > 1e-8

    def _resolve_target_lambda(
        self,
        lambda_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> float:
        if not np.any(target_mask):
            return 0.0

        if np.any(np.abs(lambda_mask[~target_mask]) > 1e-8):
            raise ValueError(
                "Non-target atoms must have lambda = 0 in this example calculator."
            )

        target_values = lambda_mask[target_mask]
        target_lambda = float(target_values[0])
        if not np.allclose(target_values, target_lambda, atol=1e-8, rtol=0.0):
            raise ValueError(
                "All selected target atoms must share the same lambda value in this example calculator."
            )
        return target_lambda

    def _predict_endpoint(
        self,
        full_atoms: Atoms,
        *,
        target_mask: np.ndarray,
        ghost_targets: bool,
    ) -> EnergyForcePrediction:
        natoms = len(full_atoms)
        full_forces = np.zeros((natoms, 3), dtype=np.float64)
        base_energy = 0.0
        base_debug: dict[str, object] = {}
        endpoint_model = (
            self._ghost_model if ghost_targets and self._ghost_model is not None else self._model
        )

        if ghost_targets:
            use_dummy_endpoint = (
                self._ghost_endpoint_mode == "dummy"
                and bool(getattr(endpoint_model, "supports_dummy_endpoint", lambda: False)())
            )
            if self._ghost_endpoint_mode == "dummy" and not use_dummy_endpoint:
                raise ValueError(
                    "ghost_endpoint_mode=`dummy` requires a model that implements "
                    "`predict_dummy_endpoint(...)`. This is currently available only "
                    "for pretrained MACE models."
                )

            if use_dummy_endpoint:
                base_prediction = endpoint_model.predict_dummy_endpoint(
                    _copy_atoms(full_atoms),
                    target_mask=target_mask,
                    properties=["energy", "forces"],
                )
                base_debug = copy.deepcopy(
                    base_prediction.get("dummy_debug", {})
                )
                base_energy = _scalar_energy(base_prediction["energy"])
                full_forces[:] = np.asarray(
                    base_prediction["forces"],
                    dtype=np.float64,
                )
            else:
                keep_mask = ~target_mask
                if np.any(keep_mask):
                    reduced_atoms = _slice_atoms(full_atoms, keep_mask)
                    base_prediction = endpoint_model.predict_one(
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
                target_mask=target_mask,
                **self._correction_kwargs,
            )
            endpoint = EnergyForcePrediction(
                energy=base_energy + correction_energy,
                forces=full_forces + correction_forces,
            )
            endpoint_details = {
                "mode": self._ghost_endpoint_mode,
                "base_energy_eV": float(base_energy),
                "lj_energy_eV": float(correction_energy),
                "d3_energy_eV": 0.0,
                "total_energy_eV": float(endpoint.energy),
                "base_force_metrics": _force_norms(full_forces, target_mask),
                "lj_force_metrics": _force_norms(correction_forces, target_mask),
                "total_force_metrics": _force_norms(endpoint.forces, target_mask),
                "target_atom_count": int(target_mask.sum()),
                "model_family": endpoint_model.family,
                "model_name": endpoint_model.model_name,
            }
            if base_debug:
                endpoint_details["dummy_debug"] = base_debug
        else:
            base_prediction = endpoint_model.predict_one(
                _copy_atoms(full_atoms),
                properties=["energy", "forces"],
            )
            base_energy = _scalar_energy(base_prediction["energy"])
            full_forces[:] = np.asarray(base_prediction["forces"], dtype=np.float64)
            endpoint = EnergyForcePrediction(
                energy=base_energy,
                forces=full_forces,
            )
            endpoint_details = {
                "mode": "real",
                "base_energy_eV": float(base_energy),
                "lj_energy_eV": 0.0,
                "d3_energy_eV": 0.0,
                "total_energy_eV": float(endpoint.energy),
                "base_force_metrics": _force_norms(full_forces, target_mask),
                "lj_force_metrics": _force_norms(
                    np.zeros_like(full_forces), target_mask
                ),
                "total_force_metrics": _force_norms(endpoint.forces, target_mask),
                "target_atom_count": int(target_mask.sum()),
                "model_family": endpoint_model.family,
                "model_name": endpoint_model.model_name,
            }

        if self._use_d3:
            d3_energy, d3_forces = ghost_target_d3_correction(
                full_atoms,
                1.0 if ghost_targets else 0.0,
                target_mask=target_mask,
                **self._d3_kwargs,
            )
            endpoint = EnergyForcePrediction(
                energy=endpoint.energy + d3_energy,
                forces=endpoint.forces + d3_forces,
            )
            endpoint_details["d3_energy_eV"] = float(d3_energy)
            endpoint_details["d3_force_metrics"] = _force_norms(d3_forces, target_mask)
            endpoint_details["total_energy_eV"] = float(endpoint.energy)
            endpoint_details["total_force_metrics"] = _force_norms(
                endpoint.forces, target_mask
            )

        if ghost_targets:
            self._last_ghost_endpoint_details = endpoint_details
        else:
            self._last_real_endpoint_details = endpoint_details

        return endpoint

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
        target_mask = self._resolve_target_mask(full_atoms, lambda_mask)
        target_lambda = self._resolve_target_lambda(lambda_mask, target_mask)

        if np.any(target_mask) and target_lambda <= 1e-8 and (
            self._fixed_target_mask is None and self._target_array_name not in full_atoms.arrays
        ):
            raise ValueError(
                "Target atoms at lambda=0 require an explicit target mask. "
                f"Provide `target_mask=...` or `atoms.arrays[{self._target_array_name!r}]`."
            )

        real_endpoint = self._predict_endpoint(
            full_atoms,
            target_mask=target_mask,
            ghost_targets=False,
        )
        ghost_endpoint = self._predict_endpoint(
            full_atoms,
            target_mask=target_mask,
            ghost_targets=True,
        )

        self._last_real_endpoint_energy = float(real_endpoint.energy)
        self._last_ghost_endpoint_energy = float(ghost_endpoint.energy)
        self.results["lambda0_energy"] = self._last_real_endpoint_energy
        self.results["lambda1_energy"] = self._last_ghost_endpoint_energy

        if not np.any(target_mask) or target_lambda <= 1e-8:
            endpoint = real_endpoint
        elif target_lambda >= 1.0 - 1e-8:
            endpoint = ghost_endpoint
        else:
            endpoint = interpolate_energy_forces(
                real_endpoint,
                ghost_endpoint,
                target_lambda,
            )

        if "energy" in requested or "free_energy" in requested:
            self.results["energy"] = float(endpoint.energy)
            self.results["free_energy"] = float(endpoint.energy)
        if "forces" in requested:
            self.results["forces"] = endpoint.forces


__all__ = ["GhostTargetCorrectionCalculator"]
