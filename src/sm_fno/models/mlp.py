"""MLP baseline for flattened PDE states."""

from __future__ import annotations

import torch
from torch import nn

from sm_fno.models.base import ForecastModel


class MLPBaseline(ForecastModel):
    """Simple MLP baseline for Heat1D pipeline validation.

    When ``grid_size`` is provided, the model flattens the input time window and
    spatial grid, then predicts a future time window. Without ``grid_size``, it
    falls back to the original pointwise channel MLP used by scaffold shape
    tests.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        input_steps: int = 1,
        pred_steps: int = 1,
        grid_size: int | None = None,
    ) -> None:
        """Initialize the baseline."""
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if input_steps < 1:
            raise ValueError("input_steps must be at least 1.")
        if pred_steps < 1:
            raise ValueError("pred_steps must be at least 1.")
        if grid_size is not None and grid_size < 1:
            raise ValueError("grid_size must be positive when provided.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.grid_size = grid_size
        input_dim = in_channels if grid_size is None else input_steps * grid_size * in_channels
        output_dim = out_channels if grid_size is None else pred_steps * grid_size * out_channels

        self.network = self._build_network(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    @staticmethod
    def _build_network(
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Sequential:
        """Build a small fully connected network."""
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict future states.

        Args:
            inputs: Either ``(batch, grid, channels)`` for pointwise mode or
                ``(batch, input_steps, grid, channels)`` for flattened
                spatiotemporal mode.
        """
        if self.grid_size is None:
            return self.network(inputs)

        if inputs.ndim != 4:
            raise ValueError(
                "Flattened MLP mode expects inputs with shape "
                "(batch, time, grid, channels)."
            )
        batch_size = inputs.shape[0]
        flat_inputs = inputs.reshape(batch_size, -1)
        outputs = self.network(flat_inputs)
        return outputs.reshape(batch_size, self.pred_steps, self.grid_size, self.out_channels)
