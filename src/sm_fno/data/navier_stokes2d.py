"""CPU-friendly 2D periodic Navier-Stokes vorticity data generation."""

from __future__ import annotations

import torch


def _make_smooth_vorticity_initial_conditions(
    num_samples: int,
    grid_size: int,
    domain_length: float,
    seed: int,
    *,
    max_modes: int = 4,
) -> torch.Tensor:
    """Create smooth zero-mean vorticity fields from low Fourier modes."""
    generator = torch.Generator().manual_seed(seed)
    coords = torch.linspace(0.0, domain_length, grid_size + 1, dtype=torch.float32)[:-1]
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    fields = torch.zeros(num_samples, grid_size, grid_size, dtype=torch.float32)

    for mode_y in range(1, max_modes + 1):
        for mode_x in range(1, max_modes + 1):
            scale = 1.0 / float(mode_x * mode_x + mode_y * mode_y)
            phase = 2.0 * torch.pi * (mode_x * x + mode_y * y) / domain_length
            sin_coeff = torch.randn(num_samples, generator=generator) * scale
            cos_coeff = torch.randn(num_samples, generator=generator) * scale
            fields = fields + sin_coeff[:, None, None] * torch.sin(phase)
            fields = fields + cos_coeff[:, None, None] * torch.cos(phase)

    fields = fields - fields.mean(dim=(1, 2), keepdim=True)
    max_abs = fields.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return fields / max_abs


def _spectral_operators(
    grid_size: int,
    domain_length: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return physical wavenumbers for periodic spectral derivatives."""
    dx = domain_length / grid_size
    frequencies = 2.0 * torch.pi * torch.fft.fftfreq(grid_size, d=dx)
    k_y = frequencies[:, None]
    k_x = frequencies[None, :]
    wave_number_squared = k_x.square() + k_y.square()
    return k_x, k_y, wave_number_squared


def _velocity_and_gradients(
    vorticity: torch.Tensor,
    *,
    k_x: torch.Tensor,
    k_y: torch.Tensor,
    wave_number_squared: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute velocity and vorticity gradients from vorticity."""
    vorticity_hat = torch.fft.fft2(vorticity, dim=(-2, -1))
    safe_wave_number_squared = wave_number_squared.clone()
    safe_wave_number_squared[0, 0] = 1.0
    stream_hat = vorticity_hat / safe_wave_number_squared
    stream_hat[:, 0, 0] = 0.0

    velocity_x = torch.fft.ifft2(1j * k_y * stream_hat, dim=(-2, -1)).real
    velocity_y = torch.fft.ifft2(-1j * k_x * stream_hat, dim=(-2, -1)).real
    grad_x = torch.fft.ifft2(1j * k_x * vorticity_hat, dim=(-2, -1)).real
    grad_y = torch.fft.ifft2(1j * k_y * vorticity_hat, dim=(-2, -1)).real
    return velocity_x, velocity_y, grad_x, grad_y


def generate_navier_stokes2d(
    num_samples: int,
    grid_size: int,
    time_steps: int,
    *,
    viscosity: float = 0.001,
    dt: float = 0.001,
    domain_length: float = 1.0,
    seed: int = 42,
    cfl_safety: float = 0.95,
    save_every: int = 1,
    initial_amplitude: float = 1.0,
    max_modes: int = 4,
) -> torch.Tensor:
    """Generate 2D viscous incompressible Navier-Stokes vorticity trajectories.

    The periodic vorticity equation is advanced with pseudo-spectral
    derivatives, explicit nonlinear advection, and semi-implicit diffusion:

    ``omega_t + u dot grad(omega) = viscosity * Laplacian(omega)``.

    ``dt`` is the internal solver time step. ``save_every`` controls the number
    of internal solver steps between saved frames, so the effective saved-frame
    spacing is ``dt * save_every``. Increasing ``save_every`` is useful for
    persistence-hard diagnostics because it makes consecutive saved frames less
    stationary while preserving the internal CFL stability check.

    This generator is intended for small CPU protocol validation, not
    high-accuracy computational fluid dynamics.

    Returns:
        Tensor with shape ``(num_samples, time_steps, grid_size, grid_size, 1)``.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least 1.")
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive.")
    if cfl_safety <= 0.0:
        raise ValueError("cfl_safety must be positive.")
    if save_every < 1:
        raise ValueError("save_every must be at least 1.")
    if initial_amplitude <= 0.0:
        raise ValueError("initial_amplitude must be positive.")
    if max_modes < 1:
        raise ValueError("max_modes must be at least 1.")

    dx = domain_length / grid_size
    k_x, k_y, wave_number_squared = _spectral_operators(grid_size, domain_length)
    current = initial_amplitude * _make_smooth_vorticity_initial_conditions(
        num_samples,
        grid_size,
        domain_length,
        seed,
        max_modes=max_modes,
    )
    trajectories = torch.empty(
        num_samples,
        time_steps,
        grid_size,
        grid_size,
        1,
        dtype=torch.float32,
    )
    trajectories[:, 0, :, :, 0] = current

    for time_index in range(1, time_steps):
        for internal_step in range(save_every):
            solver_step = (time_index - 1) * save_every + internal_step + 1
            velocity_x, velocity_y, grad_x, grad_y = _velocity_and_gradients(
                current,
                k_x=k_x,
                k_y=k_y,
                wave_number_squared=wave_number_squared,
            )
            max_speed = float(torch.sqrt(velocity_x.square() + velocity_y.square()).amax())
            cfl_number = max_speed * dt / dx
            if cfl_number > cfl_safety:
                raise ValueError(
                    "Navier-Stokes generator violates the explicit advection CFL limit "
                    f"before solver step {solver_step}: CFL={cfl_number:.4f}, "
                    f"cfl_safety={cfl_safety:.4f}."
                )

            advection = velocity_x * grad_x + velocity_y * grad_y
            current_hat = torch.fft.fft2(current, dim=(-2, -1))
            advection_hat = torch.fft.fft2(advection, dim=(-2, -1))
            next_hat = (current_hat - dt * advection_hat) / (
                1.0 + viscosity * dt * wave_number_squared
            )
            next_hat[:, 0, 0] = 0.0
            current = torch.fft.ifft2(next_hat, dim=(-2, -1)).real.to(dtype=torch.float32)
            if not torch.isfinite(current).all():
                raise FloatingPointError("Navier-Stokes generator produced non-finite values.")
        trajectories[:, time_index, :, :, 0] = current

    return trajectories
