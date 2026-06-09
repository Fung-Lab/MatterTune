from __future__ import annotations

import copy
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mattertune.alchemical import EnergyForcePrediction
from mattertune.alchemical import interpolate_energy_forces
from mattertune.corrections import ghost_target_d3_correction
from mattertune.corrections import soft_core_lj_correction

from ghost_target_calculator import _copy_atoms
from ghost_target_calculator import _force_norms
from ghost_target_calculator import _normalize_lambda_mask
from ghost_target_calculator import _normalize_target_mask
from ghost_target_calculator import _scalar_energy
from ghost_target_calculator import _slice_atoms


def _device_index(device: str) -> int | None:
    parsed = torch.device(device)
    if parsed.type != "cuda" or parsed.index is None:
        return None
    return int(parsed.index)


def _set_current_device(model: Any) -> None:
    device = getattr(model, "device", None)
    if device is None:
        return
    index = _device_index(str(device))
    if index is not None:
        torch.cuda.set_device(index)


class ParallelGhostTargetCorrectionCalculator(Calculator):
    """Dual-GPU endpoint-parallel ghost-target calculator.

    This calculator preserves the ASE-facing behavior of
    ``GhostTargetCorrectionCalculator`` while evaluating the real and ghost
    endpoints in two worker threads backed by independent model instances.
    """

    def __init__(
        self,
        model: Any,
        *,
        ghost_model: Any,
        lambda_mask: Sequence[float] | np.ndarray | None = None,
        target_mask: Sequence[bool] | np.ndarray | None = None,
        lambda_array_name: str = "alchemical_lambda",
        target_array_name: str = "alchemical_target",
        epsilon: float = 1.0,
        sigma: float = 1.0,
        smooth: bool = True,
        ghost_endpoint_mode: str = "delete",
        use_d3: bool = False,
        d3_method: str = "pbe",
        d3_damping: str = "d3bj",
    ):
        super().__init__()
        self._validate_model(model, "real-endpoint")
        self._validate_model(ghost_model, "ghost-endpoint")
        self._validate_devices(model, ghost_model)

        self._model = model
        self._ghost_model = ghost_model
        self._fixed_lambda_mask = None if lambda_mask is None else np.asarray(
            lambda_mask, dtype=np.float64
        )
        self._fixed_target_mask = None if target_mask is None else np.asarray(
            target_mask, dtype=bool
        )
        self._lambda_array_name = lambda_array_name
        self._target_array_name = target_array_name
        self._correction_kwargs = {
            "epsilon": epsilon,
            "sigma": sigma,
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
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.implemented_properties = ["energy", "forces", "free_energy"]

    @staticmethod
    def _validate_model(model: Any, label: str) -> None:
        implemented_properties = tuple(getattr(model, "implemented_properties", ()))
        if "energy" not in implemented_properties:
            raise ValueError(f"The {label} model must implement `energy`.")
        if "forces" not in implemented_properties:
            raise ValueError(f"The {label} model must implement `forces`.")

    @staticmethod
    def _validate_devices(model: Any, ghost_model: Any) -> None:
        real_device = str(getattr(model, "device", ""))
        ghost_device = str(getattr(ghost_model, "device", ""))
        real_index = _device_index(real_device)
        ghost_index = _device_index(ghost_device)
        if real_index is None or ghost_index is None:
            raise ValueError(
                "ParallelGhostTargetCorrectionCalculator requires explicit CUDA "
                f"devices such as cuda:7 and cuda:6; got {real_device!r} and "
                f"{ghost_device!r}."
            )
        if real_index == ghost_index:
            raise ValueError(
                "ParallelGhostTargetCorrectionCalculator requires distinct CUDA "
                f"devices; got {real_device!r} and {ghost_device!r}."
            )
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available for dual-GPU endpoint parallelism.")
        device_count = torch.cuda.device_count()
        if real_index >= device_count or ghost_index >= device_count:
            raise ValueError(
                f"Requested CUDA devices {real_device!r} and {ghost_device!r}, "
                f"but only {device_count} CUDA devices are visible."
            )

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

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
    ) -> tuple[EnergyForcePrediction, dict[str, object]]:
        endpoint_model = self._ghost_model if ghost_targets else self._model
        _set_current_device(endpoint_model)

        natoms = len(full_atoms)
        full_forces = np.zeros((natoms, 3), dtype=np.float64)
        base_energy = 0.0
        base_debug: dict[str, object] = {}

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
                base_debug = copy.deepcopy(base_prediction.get("dummy_debug", {}))
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

        return endpoint, endpoint_details

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

        real_future = self._executor.submit(
            self._predict_endpoint,
            full_atoms,
            target_mask=target_mask,
            ghost_targets=False,
        )
        ghost_future = self._executor.submit(
            self._predict_endpoint,
            full_atoms,
            target_mask=target_mask,
            ghost_targets=True,
        )
        real_endpoint, real_details = real_future.result()
        ghost_endpoint, ghost_details = ghost_future.result()

        self._last_real_endpoint_energy = float(real_endpoint.energy)
        self._last_ghost_endpoint_energy = float(ghost_endpoint.energy)
        self._last_real_endpoint_details = real_details
        self._last_ghost_endpoint_details = ghost_details
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

        self.results["energy"] = float(endpoint.energy)
        self.results["free_energy"] = float(endpoint.energy)
        self.results["forces"] = endpoint.forces


__all__ = ["ParallelGhostTargetCorrectionCalculator"]
