"""Transformer baseline for 2D PDE forecasting."""

from __future__ import annotations

import torch
from torch import nn

from sm_fno.models.base import ForecastModel


class Transformer2DBaseline(ForecastModel):
    """Small temporal-attention baseline for fixed-window 2D forecasting."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        input_steps: int = 1,
        pred_steps: int = 1,
    ) -> None:
        """Initialize the Transformer2D baseline."""
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
        """Predict future 2D fields.

        Args:
            inputs: Tensor with shape ``(batch, time, height, width, channels)``.

        Returns:
            Tensor with shape ``(batch, pred_time, height, width, out_channels)``.
        """
        self._validate_inputs(inputs)
        batch_size, time_steps, height, width, _ = inputs.shape

        temporal_tokens = inputs.permute(0, 2, 3, 1, 4).reshape(
            batch_size * height * width,
            time_steps,
            self.in_channels,
        )
        hidden = self.lift(temporal_tokens)
        hidden = hidden + self.position.unsqueeze(0)
        hidden = self.encoder(hidden)
        final_hidden = hidden[:, -1]
        outputs = self.project(final_hidden)
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
                "Transformer2DBaseline expects inputs with shape "
                "(batch, time, height, width, channels)."
            )
        if inputs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {inputs.shape[1]}.")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[-1]}.")
