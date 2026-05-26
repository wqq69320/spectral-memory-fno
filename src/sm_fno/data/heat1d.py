"""Utilities for generating 1D heat-equation trajectories."""

from __future__ import annotations

import warnings

import torch


def _make_smooth_initial_conditions(
    num_samples: int,
    grid_size: int,
    domain_length: float,
    seed: int,
    *,
    max_modes: int = 6,
) -> torch.Tensor:
    """Create random smooth periodic initial conditions.

    The initial fields are random low-frequency Fourier series. Coefficients are
    scaled by ``1 / k^2`` so higher modes contribute less energy, producing
    smooth fields that are suitable for a small finite-difference smoke dataset.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(0.0, domain_length, grid_size + 1, dtype=torch.float32)[:-1]
    modes = torch.arange(1, max_modes + 1, dtype=torch.float32)
    phases = 2.0 * torch.pi * x[:, None] * modes[None, :] / domain_length

    scale = 1.0 / modes.square()
    sin_coeff = torch.randn(num_samples, max_modes, generator=generator) * scale
    cos_coeff = torch.randn(num_samples, max_modes, generator=generator) * scale

    fields = torch.einsum("sm,xm->sx", sin_coeff, torch.sin(phases))
    fields = fields + torch.einsum("sm,xm->sx", cos_coeff, torch.cos(phases))
    fields = fields - fields.mean(dim=1, keepdim=True)
    max_abs = fields.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    return fields / max_abs


def generate_heat1d(
    num_samples: int,
    grid_size: int,
    time_steps: int,
    *,
    alpha: float = 0.01,
    dt: float = 0.001,
    domain_length: float = 1.0,
    seed: int = 42,
) -> torch.Tensor:
    """Generate 1D heat-equation trajectories on a periodic domain.

    The solver uses the explicit second-order finite-difference scheme

    ``u[n + 1, i] = u[n, i] + r * (u[n, i + 1] - 2u[n, i] + u[n, i - 1])``

    with periodic boundary conditions and ``r = alpha * dt / dx^2``. The
    explicit method is stable for this equation when ``r <= 0.5``. This function
    raises ``ValueError`` when that condition is violated and emits a warning
    when the setting is close to the limit.

    Returns:
        Tensor with shape ``(num_samples, time_steps, grid_size, 1)``.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3 for periodic finite differences.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least 1.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive.")

    dx = domain_length / grid_size
    stability_number = alpha * dt / (dx * dx)
    if stability_number > 0.5:
        raise ValueError(
            "Explicit heat solver is unstable: "
            f"alpha * dt / dx^2 = {stability_number:.4f} > 0.5."
        )
    if stability_number > 0.45:
        warnings.warn(
            "Explicit heat solver is close to the stability limit "
            f"(alpha * dt / dx^2 = {stability_number:.4f}).",
            stacklevel=2,
        )

    trajectories = torch.empty(num_samples, time_steps, grid_size, 1, dtype=torch.float32)
    current = _make_smooth_initial_conditions(num_samples, grid_size, domain_length, seed)
    trajectories[:, 0, :, 0] = current

    for time_index in range(1, time_steps):
        laplacian = torch.roll(current, shifts=-1, dims=1) - 2.0 * current
        laplacian = laplacian + torch.roll(current, shifts=1, dims=1)
        current = current + stability_number * laplacian
        trajectories[:, time_index, :, 0] = current

    return trajectories
