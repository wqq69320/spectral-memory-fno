"""Spectral Memory Fourier Neural Operator for 1D fields."""

from __future__ import annotations

import torch
from torch import nn

from sm_fno.models.base import ForecastModel
from sm_fno.models.fno1d import FNOBlock1D
from sm_fno.models.ssm import DiagonalSSM


class SpectralMemoryFNO1D(ForecastModel):
    """SM-FNO model for fixed-window 1D PDE forecasting.

    Each input state is first mixed across space with FNO blocks. For every grid
    point, a diagonal SSM then propagates temporal memory over the input window.
    The final memory state is projected to the configured prediction horizon.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 16,
        width: int = 64,
        state_dim: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        input_steps: int = 1,
        pred_steps: int = 1,
    ) -> None:
        """Initialize the SM-FNO model."""
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
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList(FNOBlock1D(width, modes) for _ in range(depth))
        self.memory = DiagonalSSM(input_dim=width, state_dim=state_dim, output_dim=width)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, pred_steps * out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict future 1D fields.

        Args:
            inputs: Tensor with shape ``(batch, time, grid, channels)``.

        Returns:
            Tensor with shape ``(batch, pred_time, grid, out_channels)``.
        """
        inputs, legacy_one_step = self._prepare_inputs(inputs)
        batch_size, time_steps, grid_size, _ = inputs.shape

        hidden = self.lift(inputs)
        hidden = hidden.reshape(batch_size * time_steps, grid_size, -1).permute(0, 2, 1)

        for block in self.blocks:
            hidden = self.dropout(block(hidden))

        hidden_grid = hidden.permute(0, 2, 1).reshape(batch_size, time_steps, grid_size, -1)
        temporal_inputs = hidden_grid.permute(0, 2, 1, 3).reshape(
            batch_size * grid_size,
            time_steps,
            -1,
        )
        temporal_outputs, _ = self.memory(temporal_inputs)
        final_memory = temporal_outputs[:, -1].reshape(batch_size, grid_size, -1)

        outputs = self.project(final_memory)
        outputs = outputs.reshape(batch_size, grid_size, self.pred_steps, self.out_channels)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        if legacy_one_step:
            return outputs[:, 0]
        return outputs

    def _prepare_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Validate supported input shapes and normalize to a 4D window."""
        legacy_one_step = False
        if inputs.ndim == 3:
            if self.input_steps != 1:
                raise ValueError(
                    "Legacy SM-FNO inputs with shape (batch, grid, channels) "
                    "require input_steps=1."
                )
            inputs = inputs.unsqueeze(1)
            legacy_one_step = self.pred_steps == 1
        elif inputs.ndim != 4:
            raise ValueError(
                "SpectralMemoryFNO1D expects inputs with shape "
                "(batch, time, grid, channels) or legacy shape (batch, grid, channels)."
            )

        if inputs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {inputs.shape[1]}.")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[-1]}.")
        return inputs, legacy_one_step
