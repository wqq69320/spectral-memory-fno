"""Spectral Memory Fourier Neural Operator for 2D fields."""

from __future__ import annotations

import torch
from torch import nn

from sm_fno.models.base import ForecastModel
from sm_fno.models.fno2d import FNO2D, FNOBlock2D
from sm_fno.models.ssm import DiagonalSSM, StableGatedDiagonalSSM


class SpectralMemoryFNO2D(ForecastModel):
    """SM-FNO model for fixed-window 2D PDE forecasting."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 8,
        width: int = 32,
        state_dim: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        input_steps: int = 1,
        pred_steps: int = 1,
    ) -> None:
        """Initialize the 2D SM-FNO model."""
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
        self.blocks = nn.ModuleList(FNOBlock2D(width, modes) for _ in range(depth))
        self.memory = DiagonalSSM(input_dim=width, state_dim=state_dim, output_dim=width)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, pred_steps * out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict future 2D fields.

        Args:
            inputs: Tensor with shape ``(batch, time, height, width, channels)``.

        Returns:
            Tensor with shape ``(batch, pred_time, height, width, out_channels)``.
        """
        self._validate_inputs(inputs)
        batch_size, time_steps, height, width, _ = inputs.shape

        hidden = self.lift(inputs)
        hidden = hidden.reshape(batch_size * time_steps, height, width, -1).permute(0, 3, 1, 2)
        for block in self.blocks:
            hidden = self.dropout(block(hidden))

        hidden_grid = hidden.permute(0, 2, 3, 1).reshape(
            batch_size,
            time_steps,
            height,
            width,
            -1,
        )
        temporal_inputs = hidden_grid.permute(0, 2, 3, 1, 4).reshape(
            batch_size * height * width,
            time_steps,
            -1,
        )
        temporal_outputs, _ = self.memory(temporal_inputs)
        final_memory = temporal_outputs[:, -1].reshape(batch_size, height, width, -1)

        outputs = self.project(final_memory)
        outputs = outputs.reshape(
            batch_size,
            height,
            width,
            self.pred_steps,
            self.out_channels,
        )
        return outputs.permute(0, 3, 1, 2, 4).contiguous()

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """Validate 2D input windows."""
        if inputs.ndim != 5:
            raise ValueError(
                "SpectralMemoryFNO2D expects inputs with shape "
                "(batch, time, height, width, channels)."
            )
        if inputs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {inputs.shape[1]}.")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[-1]}.")


class SpectralMemoryFNO2DV2(ForecastModel):
    """SM-FNO2D v2 with stable gated temporal memory."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 8,
        width: int = 32,
        state_dim: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        input_steps: int = 1,
        pred_steps: int = 1,
    ) -> None:
        """Initialize the v2 2D SM-FNO model."""
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
        self.blocks = nn.ModuleList(FNOBlock2D(width, modes) for _ in range(depth))
        self.memory = StableGatedDiagonalSSM(
            input_dim=width,
            state_dim=state_dim,
            output_dim=width,
        )
        self.temporal_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, pred_steps * out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict future 2D fields with stable gated temporal memory."""
        self._validate_inputs(inputs)
        batch_size, time_steps, height, width, _ = inputs.shape

        hidden = self.lift(inputs)
        hidden = hidden.reshape(batch_size * time_steps, height, width, -1).permute(0, 3, 1, 2)
        for block in self.blocks:
            hidden = hidden + self.dropout(block(hidden))

        hidden_grid = hidden.permute(0, 2, 3, 1).reshape(
            batch_size,
            time_steps,
            height,
            width,
            -1,
        )
        temporal_inputs = hidden_grid.permute(0, 2, 3, 1, 4).reshape(
            batch_size * height * width,
            time_steps,
            -1,
        )
        temporal_outputs, _ = self.memory(temporal_inputs)
        final_memory = self.temporal_norm(temporal_outputs[:, -1])
        final_memory = final_memory.reshape(batch_size, height, width, -1)

        outputs = self.project(final_memory)
        outputs = outputs.reshape(
            batch_size,
            height,
            width,
            self.pred_steps,
            self.out_channels,
        )
        return outputs.permute(0, 3, 1, 2, 4).contiguous()

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """Validate 2D input windows."""
        if inputs.ndim != 5:
            raise ValueError(
                "SpectralMemoryFNO2DV2 expects inputs with shape "
                "(batch, time, height, width, channels)."
            )
        if inputs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {inputs.shape[1]}.")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[-1]}.")


class SpectralMemoryFNO2DV3(ForecastModel):
    """SM-FNO2D v3 with an FNO residual base and conservative SSM correction."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 8,
        width: int = 32,
        state_dim: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        input_steps: int = 1,
        pred_steps: int = 1,
        gate_limit: float = 0.25,
        gate_bias: float = -4.0,
        correction_scale: float = 1.0,
    ) -> None:
        """Initialize the v3 2D SM-FNO model."""
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if input_steps < 1:
            raise ValueError("input_steps must be at least 1.")
        if pred_steps < 1:
            raise ValueError("pred_steps must be at least 1.")
        if gate_limit <= 0.0:
            raise ValueError("gate_limit must be positive.")
        if correction_scale <= 0.0:
            raise ValueError("correction_scale must be positive.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.gate_limit = gate_limit
        self.correction_scale = correction_scale

        self.base = FNO2D(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            width=width,
            depth=depth,
            dropout=dropout,
            input_steps=input_steps,
            pred_steps=pred_steps,
        )
        self.temporal_lift = nn.Linear(in_channels, width)
        self.memory = StableGatedDiagonalSSM(
            input_dim=width,
            state_dim=state_dim,
            output_dim=width,
        )
        self.temporal_norm = nn.LayerNorm(width)
        self.correction_project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, pred_steps * out_channels),
        )
        self.gate_project = nn.Linear(width, pred_steps * out_channels)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.gate_project.weight)
        nn.init.constant_(self.gate_project.bias, gate_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict future 2D fields as FNO base plus gated SSM correction."""
        self._validate_inputs(inputs)
        base_prediction = self.base(inputs)
        batch_size, time_steps, height, width, _ = inputs.shape

        hidden = self.dropout(self.temporal_lift(inputs))
        temporal_inputs = hidden.permute(0, 2, 3, 1, 4).reshape(
            batch_size * height * width,
            time_steps,
            -1,
        )
        temporal_outputs, _ = self.memory(temporal_inputs)
        final_memory = self.temporal_norm(temporal_outputs[:, -1]).reshape(
            batch_size,
            height,
            width,
            -1,
        )

        correction = self.correction_project(final_memory).reshape(
            batch_size,
            height,
            width,
            self.pred_steps,
            self.out_channels,
        )
        gate = self.gate_limit * torch.sigmoid(self.gate_project(final_memory))
        gate = gate.reshape(
            batch_size,
            height,
            width,
            self.pred_steps,
            self.out_channels,
        )
        correction = correction.permute(0, 3, 1, 2, 4).contiguous()
        gate = gate.permute(0, 3, 1, 2, 4).contiguous()
        return base_prediction + self.correction_scale * gate * correction

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """Validate 2D input windows."""
        if inputs.ndim != 5:
            raise ValueError(
                "SpectralMemoryFNO2DV3 expects inputs with shape "
                "(batch, time, height, width, channels)."
            )
        if inputs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {inputs.shape[1]}.")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[-1]}.")
