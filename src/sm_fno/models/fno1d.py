"""1D Fourier Neural Operator components."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sm_fno.models.base import ForecastModel


class SpectralConv1D(nn.Module):
    """1D spectral convolution over retained Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        """Initialize learned complex spectral weights."""
        super().__init__()
        if modes < 1:
            raise ValueError("modes must be at least 1.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            inputs: Tensor with shape ``(batch, channels, grid)``.

        Returns:
            Tensor with shape ``(batch, out_channels, grid)``.
        """
        if inputs.ndim != 3:
            raise ValueError("SpectralConv1D expects inputs with shape (batch, channels, grid).")

        batch_size, _, grid_size = inputs.shape
        inputs_ft = torch.fft.rfft(inputs, dim=-1)
        output_ft = torch.zeros(
            batch_size,
            self.out_channels,
            inputs_ft.shape[-1],
            device=inputs.device,
            dtype=inputs_ft.dtype,
        )
        retained_modes = min(self.modes, inputs_ft.shape[-1])
        output_ft[:, :, :retained_modes] = torch.einsum(
            "bix,iox->box",
            inputs_ft[:, :, :retained_modes],
            self.weight[:, :, :retained_modes].to(dtype=inputs_ft.dtype),
        )
        return torch.fft.irfft(output_ft, n=grid_size, dim=-1)


class FNOBlock1D(nn.Module):
    """Single FNO block with spectral and pointwise paths."""

    def __init__(self, width: int, modes: int) -> None:
        """Initialize the block."""
        super().__init__()
        self.spectral = SpectralConv1D(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply spectral and pointwise updates."""
        return F.gelu(self.spectral(inputs) + self.pointwise(inputs))


class FNO1D(ForecastModel):
    """1D FNO baseline for fixed-window Heat1D forecasting.

    The research pipeline uses inputs with shape ``(batch, time, grid,
    channels)`` and outputs with shape ``(batch, pred_time, grid, channels)``.
    A legacy ``(batch, grid, channels)`` one-step mode is retained for the
    repository's smoke-test entry point.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 16,
        width: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        input_steps: int = 1,
        pred_steps: int = 1,
    ) -> None:
        """Initialize the 1D FNO."""
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if input_steps < 1:
            raise ValueError("input_steps must be at least 1.")
        if pred_steps < 1:
            raise ValueError("pred_steps must be at least 1.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.lift = nn.Linear(input_steps * in_channels, width)
        self.blocks = nn.ModuleList(FNOBlock1D(width, modes) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, pred_steps * out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict future 1D fields."""
        features, legacy_one_step = self._prepare_inputs(inputs)
        batch_size, grid_size, _ = features.shape

        hidden = self.lift(features).permute(0, 2, 1)
        for block in self.blocks:
            hidden = self.dropout(block(hidden))
        hidden = hidden.permute(0, 2, 1)
        outputs = self.project(hidden)
        outputs = outputs.reshape(batch_size, grid_size, self.pred_steps, self.out_channels)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        if legacy_one_step:
            return outputs[:, 0]
        return outputs

    def _prepare_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Convert supported input shapes to ``(batch, grid, time * channels)``."""
        legacy_one_step = False
        if inputs.ndim == 3:
            if self.input_steps != 1:
                raise ValueError(
                    "Legacy FNO1D inputs with shape (batch, grid, channels) "
                    "require input_steps=1."
                )
            inputs = inputs.unsqueeze(1)
            legacy_one_step = self.pred_steps == 1
        elif inputs.ndim != 4:
            raise ValueError(
                "FNO1D expects inputs with shape (batch, time, grid, channels) "
                "or legacy shape (batch, grid, channels)."
            )

        if inputs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {inputs.shape[1]}.")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[-1]}.")

        batch_size, time_steps, grid_size, channels = inputs.shape
        features = inputs.permute(0, 2, 1, 3).reshape(
            batch_size,
            grid_size,
            time_steps * channels,
        )
        return features, legacy_one_step


SpectralConv1d = SpectralConv1D
