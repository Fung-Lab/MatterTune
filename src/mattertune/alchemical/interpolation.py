from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EnergyForcePrediction:
    energy: float
    forces: NDArray[np.float64]


def validate_lambda_value(lambda_value: float) -> float:
    value = float(lambda_value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"lambda_value must lie in [0, 1], got {value}.")
    return value


def interpolate_scalar(
    real_value: float,
    ghost_value: float,
    lambda_value: float,
) -> float:
    lam = validate_lambda_value(lambda_value)
    return (1.0 - lam) * float(real_value) + lam * float(ghost_value)


def interpolate_array(
    real_value: NDArray[np.float64],
    ghost_value: NDArray[np.float64],
    lambda_value: float,
) -> NDArray[np.float64]:
    lam = validate_lambda_value(lambda_value)
    return (1.0 - lam) * real_value + lam * ghost_value


def interpolate_energy_forces(
    real_prediction: EnergyForcePrediction,
    ghost_prediction: EnergyForcePrediction,
    lambda_value: float,
) -> EnergyForcePrediction:
    return EnergyForcePrediction(
        energy=interpolate_scalar(
            real_prediction.energy,
            ghost_prediction.energy,
            lambda_value,
        ),
        forces=interpolate_array(
            real_prediction.forces,
            ghost_prediction.forces,
            lambda_value,
        ),
    )
