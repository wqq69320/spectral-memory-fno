"""Utilities for generating 1D viscous Burgers-equation trajectories."""

from __future__ import annotations

import warnings

import torch


def _make_smooth_initial_conditions(
    num_samples: int,
    grid_size: int,
    domain_length: float,
    seed: int,
    *,
    max_modes: int = 5,
) -> torch.Tensor:
    """Create smooth periodic initial conditions from low Fourier modes."""
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


def _check_combined_stability(
    *,
    cfl_number: float,
    diffusion_number: float,
    combined_number: float,
    stability_safety: float,
    step_label: str,
) -> None:
    """Raise if the explicit convection-diffusion update is unstable."""
    if combined_number <= stability_safety:
        return
    raise ValueError(
        "Explicit Burgers solver violates the combined "
        f"convection-diffusion stability limit before {step_label}: "
        f"CFL={cfl_number:.4f}, "
        f"diffusion={diffusion_number:.4f}, "
        f"combined={combined_number:.4f}, "
        f"stability_safety={stability_safety:.4f}."
    )


def generate_burgers1d(
    num_samples: int,
    grid_size: int,
    time_steps: int,
    *,
    viscosity: float = 0.01,
    dt: float = 0.001,
    domain_length: float = 1.0,
    seed: int = 42,
    stability_safety: float = 0.95,
) -> torch.Tensor:
    """Generate 1D viscous Burgers trajectories on a periodic domain.

    The solver uses a conservative Rusanov flux for the nonlinear advection
    term and a centered finite-difference diffusion term:

    ``u_t + (0.5 * u^2)_x = viscosity * u_xx``.

    This explicit scheme is intended for CPU-friendly protocol validation, not
    high-accuracy production simulation. It checks conservative CFL, diffusion,
    and combined explicit convection-diffusion stability limits.

    Returns:
        Tensor with shape ``(num_samples, time_steps, grid_size, 1)``.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4 for periodic finite differences.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least 1.")
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive.")
    if stability_safety <= 0.0 or stability_safety > 1.0:
        raise ValueError("stability_safety must be in the interval (0, 1].")

    dx = domain_length / grid_size
    initial = _make_smooth_initial_conditions(num_samples, grid_size, domain_length, seed)
    max_speed = float(initial.abs().amax())
    cfl_number = dt * max_speed / dx
    diffusion_number = viscosity * dt / (dx * dx)
    combined_number = cfl_number + 2.0 * diffusion_number
    _check_combined_stability(
        cfl_number=cfl_number,
        diffusion_number=diffusion_number,
        combined_number=combined_number,
        stability_safety=stability_safety,
        step_label="the first update",
    )
    if cfl_number > 1.0:
        raise ValueError(
            "Explicit Burgers solver violates the advective CFL limit: "
            f"dt * max|u| / dx = {cfl_number:.4f} > 1.0."
        )
    if diffusion_number > 0.5:
        raise ValueError(
            "Explicit Burgers solver violates the diffusion stability limit: "
            f"viscosity * dt / dx^2 = {diffusion_number:.4f} > 0.5."
        )
    if combined_number <= stability_safety and (
        cfl_number > 0.8
        or diffusion_number > 0.45
        or combined_number > 0.9 * stability_safety
    ):
        warnings.warn(
            "Burgers solver is close to an explicit stability limit "
            f"(CFL={cfl_number:.4f}, diffusion={diffusion_number:.4f}, "
            f"combined={combined_number:.4f}, safety={stability_safety:.4f}).",
            stacklevel=2,
        )

    trajectories = torch.empty(num_samples, time_steps, grid_size, 1, dtype=torch.float32)
    current = initial
    trajectories[:, 0, :, 0] = current

    for time_index in range(1, time_steps):
        max_speed = float(current.abs().amax())
        cfl_number = dt * max_speed / dx
        combined_number = cfl_number + 2.0 * diffusion_number
        _check_combined_stability(
            cfl_number=cfl_number,
            diffusion_number=diffusion_number,
            combined_number=combined_number,
            stability_safety=stability_safety,
            step_label=f"step {time_index}",
        )
        right = torch.roll(current, shifts=-1, dims=1)
        flux = 0.5 * current.square()
        right_flux = 0.5 * right.square()
        local_speed = torch.maximum(current.abs(), right.abs())
        interface_flux = 0.5 * (flux + right_flux) - 0.5 * local_speed * (right - current)
        flux_difference = interface_flux - torch.roll(interface_flux, shifts=1, dims=1)
        laplacian = right - 2.0 * current + torch.roll(current, shifts=1, dims=1)
        current = current - (dt / dx) * flux_difference + diffusion_number * laplacian
        trajectories[:, time_index, :, 0] = current

    return trajectories
