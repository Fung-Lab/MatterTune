from __future__ import annotations

import torch


def voigt_6_to_full_3x3_stress_torch(stress_vector: torch.Tensor) -> torch.Tensor:
    """
    Form a 3x3 stress matrix from a 6 component vector in Voigt notation

    Args:
        stress_vector: Tensor of shape (B, 6) where B is the batch size

    Returns:
        Tensor of shape (B, 3, 3)
    """
    # Unpack the components
    s1, s2, s3, s4, s5, s6 = stress_vector.unbind(dim=1)

    # Stack the components into a 3x3 matrix
    # Each s_i is of shape (B,)
    stress_matrix = torch.stack(
        [
            torch.stack([s1, s6, s5], dim=1),
            torch.stack([s6, s2, s4], dim=1),
            torch.stack([s5, s4, s3], dim=1),
        ],
        dim=1,
    )

    return stress_matrix
