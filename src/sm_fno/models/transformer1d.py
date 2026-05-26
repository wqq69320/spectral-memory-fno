"""Transformer baseline for 1D PDE forecasting."""

from __future__ import annotations

import torch
from torch import nn

from sm_fno.models.base import ForecastModel


class Transformer1DBaseline(ForecastModel):
    """Small temporal-attention baseline for fixed-window Heat1D forecasting.

    The main research pipeline uses inputs with shape ``(batch, time, grid,
    channels)`` and outputs with shape ``(batch, pred_time, grid, channels)``.
    For each grid point, the model applies self-attention over the input time
    window and projects the final temporal representation to the prediction
    horizon. A legacy ``(batch, grid, channels)`` one-step mode is retained for
    lightweight smoke tests.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        input_steps: int = 1,
        pred_steps: int = 1,
    ) -> None:
        """Initialize the Transformer baseline."""
        super().__init__()
        if input_steps < 1:
            raise ValueError("input_steps must be at least 1.")
        if pred_steps < 1:
            raise ValueError("pred_steps must be at least 1.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.lift = nn.Linear(in_channels, d_model)
        self.position = nn.Parameter(torch.zeros(input_steps, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.project = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, pred_steps * out_channels),
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

        temporal_tokens = inputs.permute(0, 2, 1, 3).reshape(
            batch_size * grid_size,
            time_steps,
            self.in_channels,
        )
        hidden = self.lift(temporal_tokens)
        hidden = hidden + self.position.unsqueeze(0)
        hidden = self.encoder(hidden)
        final_hidden = hidden[:, -1]
        outputs = self.project(final_hidden)
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
                    "Legacy Transformer1D inputs with shape (batch, grid, channels) "
                    "require input_steps=1."
                )
            inputs = inputs.unsqueeze(1)
            legacy_one_step = self.pred_steps == 1
        elif inputs.ndim != 4:
            raise ValueError(
                "Transformer1DBaseline expects inputs with shape "
                "(batch, time, grid, channels) or legacy shape (batch, grid, channels)."
            )

        if inputs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {inputs.shape[1]}.")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[-1]}.")
        return inputs, legacy_one_step
