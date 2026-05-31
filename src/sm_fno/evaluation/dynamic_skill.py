"""Persistence and dynamic-skill diagnostics for rollout forecasts."""

from __future__ import annotations

import torch


def persistence_forecast(inputs: torch.Tensor, steps: int) -> torch.Tensor:
    """Repeat the last input frame over a rollout horizon.

    Args:
        inputs: Tensor with shape ``(batch, input_steps, *spatial, channels)``.
        steps: Number of rollout steps to repeat.

    Returns:
        Tensor with shape ``(batch, steps, *spatial, channels)``.
    """
    if inputs.ndim < 4:
        raise ValueError("inputs must have shape (batch, input_steps, *spatial, channels).")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    last_frame = inputs[:, -1:]
    repeat_shape = [1, steps, *([1] * (inputs.ndim - 2))]
    return last_frame.repeat(*repeat_shape)


def per_timestep_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute relative L2 error for each rollout step."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}."
        )
    if target.ndim < 4:
        raise ValueError("target must have shape (batch, time, *spatial, channels).")
    reduce_dims = tuple(range(2, target.ndim))
    numerator = torch.linalg.vector_norm(prediction - target, ord=2, dim=reduce_dims)
    denominator = torch.linalg.vector_norm(target, ord=2, dim=reduce_dims)
    return torch.mean(numerator / (denominator + eps), dim=0)


def true_step_change_relative_l2(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Measure true trajectory change from the previous true frame.

    Step 1 is compared against the last input frame. Later steps are compared
    against the previous target frame.
    """
    if inputs.ndim != targets.ndim:
        raise ValueError("inputs and targets must have matching rank.")
    if inputs.shape[0] != targets.shape[0] or inputs.shape[2:] != targets.shape[2:]:
        raise ValueError("inputs and targets must share batch, spatial, and channel dims.")
    previous = torch.cat([inputs[:, -1:], targets[:, :-1]], dim=1)
    return per_timestep_relative_l2(previous, targets, eps=eps)


def delta_prediction_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare predicted and true change relative to a reference frame.

    ``reference`` is usually the last input frame and may have shape
    ``(batch, *spatial, channels)`` or ``(batch, 1, *spatial, channels)``.
    """
    if reference.ndim == target.ndim - 1:
        reference = reference.unsqueeze(1)
    if reference.ndim != target.ndim:
        raise ValueError("reference must be broadcastable over the rollout time dimension.")
    if reference.shape[0] != target.shape[0] or reference.shape[2:] != target.shape[2:]:
        raise ValueError("reference must share batch, spatial, and channel dims with target.")
    true_delta = target - reference
    predicted_delta = prediction - reference
    reduce_dims = tuple(range(2, target.ndim))
    numerator = torch.linalg.vector_norm(predicted_delta - true_delta, ord=2, dim=reduce_dims)
    denominator = torch.linalg.vector_norm(true_delta, ord=2, dim=reduce_dims)
    return torch.mean(numerator / (denominator + eps), dim=0)


def persistence_skill_score(
    model_error: torch.Tensor,
    persistence_error: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return skill relative to persistence.

    Positive values mean lower error than persistence. Zero is parity with
    persistence. Negative values mean worse error than persistence.
    """
    if model_error.shape != persistence_error.shape:
        raise ValueError("model_error and persistence_error must have the same shape.")
    return 1.0 - model_error / (persistence_error + eps)

