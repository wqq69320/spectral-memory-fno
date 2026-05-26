"""2D Fourier Neural Operator components."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sm_fno.models.base import ForecastModel


class SpectralConv2D(nn.Module):
    """2D spectral convolution over retained Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        """Initialize learned complex spectral weights."""
        super().__init__()
        if modes < 1:
            raise ValueError("modes must be at least 1.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_top = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weight_bottom = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def retained_mode_counts(self, height: int, width_rfft: int) -> tuple[int, int, int]:
        """Return non-overlapping retained mode counts for y-top, y-bottom, and x."""
        if height < 1 or width_rfft < 1:
            raise ValueError("height and width_rfft must be positive.")
        retained_top_y = min(self.modes, (height + 1) // 2)
        retained_bottom_y = min(self.modes, height - retained_top_y)
        retained_x = min(self.modes, width_rfft)
        return retained_top_y, retained_bottom_y, retained_x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            inputs: Tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Tensor with shape ``(batch, out_channels, height, width)``.
        """
        if inputs.ndim != 4:
            raise ValueError(
                "SpectralConv2D expects inputs with shape "
                "(batch, channels, height, width)."
            )

        batch_size, _, height, width = inputs.shape
        inputs_ft = torch.fft.rfft2(inputs, dim=(-2, -1))
        output_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            inputs_ft.shape[-1],
            device=inputs.device,
            dtype=inputs_ft.dtype,
        )
        retained_top_y, retained_bottom_y, retained_x = self.retained_mode_counts(
            height,
            inputs_ft.shape[-1],
        )
        output_ft[:, :, :retained_top_y, :retained_x] = torch.einsum(
            "bixy,ioxy->boxy",
            inputs_ft[:, :, :retained_top_y, :retained_x],
            self.weight_top[:, :, :retained_top_y, :retained_x].to(dtype=inputs_ft.dtype),
        )
        if retained_bottom_y > 0:
            output_ft[:, :, -retained_bottom_y:, :retained_x] = torch.einsum(
                "bixy,ioxy->boxy",
                inputs_ft[:, :, -retained_bottom_y:, :retained_x],
                self.weight_bottom[:, :, :retained_bottom_y, :retained_x].to(
                    dtype=inputs_ft.dtype
                ),
            )
        return torch.fft.irfft2(output_ft, s=(height, width), dim=(-2, -1))


class FNOBlock2D(nn.Module):
    """Single 2D FNO block with spectral and pointwise paths."""

    def __init__(self, width: int, modes: int) -> None:
        """Initialize the block."""
        super().__init__()
        self.spectral = SpectralConv2D(width, width, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply spectral and pointwise updates."""
        return F.gelu(self.spectral(inputs) + self.pointwise(inputs))


class FNO2D(ForecastModel):
    """2D FNO baseline for fixed-window vorticity forecasting."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 8,
        width: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        input_steps: int = 1,
        pred_steps: int = 1,
    ) -> None:
        """Initialize the 2D FNO."""
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
        self.blocks = nn.ModuleList(FNOBlock2D(width, modes) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, pred_steps * out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict future 2D fields."""
        features = self._prepare_inputs(inputs)
        batch_size, height, width, _ = features.shape

        hidden = self.lift(features).permute(0, 3, 1, 2)
        for block in self.blocks:
            hidden = self.dropout(block(hidden))
        hidden = hidden.permute(0, 2, 3, 1)
        outputs = self.project(hidden)
        outputs = outputs.reshape(
            batch_size,
            height,
            width,
            self.pred_steps,
            self.out_channels,
        )
        return outputs.permute(0, 3, 1, 2, 4).contiguous()

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convert inputs to ``(batch, height, width, time * channels)``."""
        if inputs.ndim != 5:
            raise ValueError(
                "FNO2D expects inputs with shape "
                "(batch, time, height, width, channels)."
            )
        if inputs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {inputs.shape[1]}.")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[-1]}.")

        batch_size, time_steps, height, width, channels = inputs.shape
        return inputs.permute(0, 2, 3, 1, 4).reshape(
            batch_size,
            height,
            width,
            time_steps * channels,
        )
