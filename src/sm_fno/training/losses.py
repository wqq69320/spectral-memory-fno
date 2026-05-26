"""Training losses."""

from __future__ import annotations

import torch


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error."""
    return torch.mean((prediction - target).square())


def relative_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute relative L2 loss over all non-batch dimensions."""
    numerator = torch.linalg.vector_norm(
        prediction - target,
        ord=2,
        dim=tuple(range(1, target.ndim)),
    )
    denominator = torch.linalg.vector_norm(target, ord=2, dim=tuple(range(1, target.ndim)))
    return torch.mean(numerator / (denominator + eps))
