"""Autoregressive rollout utilities."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def autoregressive_rollout(
    model: nn.Module,
    initial_state: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Roll out a fixed-window model for multiple future steps.

    Args:
        model: Model mapping ``(batch, time, grid, channels)`` to one or more
            future states. Legacy one-step ``(batch, grid, channels)`` models
            are also supported.
        initial_state: Initial window with shape ``(batch, time, *spatial,
            channels)`` or a legacy 1D single state with shape
            ``(batch, grid, channels)``.
        steps: Number of forecast steps.

    Returns:
        Tensor with shape ``(batch, steps, *spatial, channels)``.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    if initial_state.ndim == 3:
        current_window = initial_state.unsqueeze(1)
        legacy_one_step = True
    elif initial_state.ndim >= 4:
        current_window = initial_state
        legacy_one_step = False
    else:
        raise ValueError(
            "initial_state must have shape (batch, grid, channels) or "
            "(batch, time, grid, channels)."
        )

    predictions: list[torch.Tensor] = []
    while len(predictions) < steps:
        model_inputs = current_window[:, -1] if legacy_one_step else current_window
        next_predictions = model(model_inputs)
        if next_predictions.ndim == current_window.ndim - 1:
            next_predictions = next_predictions.unsqueeze(1)
        elif next_predictions.ndim != current_window.ndim:
            raise ValueError(
                "model must return shape (batch, *spatial, channels) or "
                "(batch, pred_time, *spatial, channels)."
            )

        for time_index in range(next_predictions.shape[1]):
            next_step = next_predictions[:, time_index : time_index + 1]
            predictions.append(next_step[:, 0])
            current_window = torch.cat([current_window[:, 1:], next_step], dim=1)
            if len(predictions) == steps:
                break

    return torch.stack(predictions, dim=1)
