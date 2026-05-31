"""Evaluation metrics for PDE forecasts."""

from __future__ import annotations

import torch


def mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error."""
    return torch.mean((prediction - target).square())


def relative_l2_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean relative L2 error over all non-batch dimensions."""
    numerator = torch.linalg.vector_norm(
        prediction - target,
        ord=2,
        dim=tuple(range(1, target.ndim)),
    )
    denominator = torch.linalg.vector_norm(target, ord=2, dim=tuple(range(1, target.ndim)))
    return torch.mean(numerator / (denominator + eps))


def per_timestep_relative_l2_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean relative L2 error for each forecast time step."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}."
        )
    if prediction.ndim < 4:
        raise ValueError(
            "per_timestep_relative_l2_error expects shape "
            "(batch, time, ..., channels)."
        )
    reduce_dims = tuple(dim for dim in range(2, target.ndim))
    numerator = torch.linalg.vector_norm(prediction - target, ord=2, dim=reduce_dims)
    denominator = torch.linalg.vector_norm(target, ord=2, dim=reduce_dims)
    return torch.mean(numerator / (denominator + eps), dim=0)


def energy_error(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute relative error in mean squared field energy."""
    pred_energy = torch.mean(prediction.square(), dim=tuple(range(1, prediction.ndim)))
    target_energy = torch.mean(target.square(), dim=tuple(range(1, target.ndim)))
    return torch.mean(torch.abs(pred_energy - target_energy) / (torch.abs(target_energy) + eps))


def fourier_spectrum_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute relative error between spatial Fourier amplitude spectra."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}."
        )
    if prediction.ndim < 3:
        raise ValueError("fourier_spectrum_error expects at least batch, spatial, channel dims.")
    spatial_dims = (
        tuple(range(2, prediction.ndim - 1))
        if prediction.ndim >= 4
        else (prediction.ndim - 2,)
    )
    pred_spectrum = torch.abs(torch.fft.rfftn(prediction, dim=spatial_dims))
    target_spectrum = torch.abs(torch.fft.rfftn(target, dim=spatial_dims))
    numerator = torch.linalg.vector_norm(
        pred_spectrum - target_spectrum,
        ord=2,
        dim=tuple(range(1, pred_spectrum.ndim)),
    )
    denominator = torch.linalg.vector_norm(
        target_spectrum,
        ord=2,
        dim=tuple(range(1, target_spectrum.ndim)),
    )
    return torch.mean(numerator / (denominator + eps))
