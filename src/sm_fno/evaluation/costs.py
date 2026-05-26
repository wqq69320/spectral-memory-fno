"""Lightweight compute-cost helpers for protocol-scale evaluations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and frozen parameters for a PyTorch model."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "model_parameter_count": int(total),
        "model_trainable_parameter_count": int(trainable),
        "model_frozen_parameter_count": int(total - trainable),
    }


def seconds_per_unit(seconds: float | None, denominator: int | float) -> float | None:
    """Return wall-clock seconds per unit, preserving missing or invalid values."""
    if seconds is None or denominator <= 0:
        return None
    return float(seconds) / float(denominator)


def inference_timing_summary(
    *,
    seconds: float | None,
    examples: int,
    steps: int | None = None,
) -> dict[str, Any]:
    """Compute normalized inference timing fields from elapsed seconds."""
    summary: dict[str, Any] = {
        "inference_seconds": seconds,
        "inference_examples_per_second": (
            float(examples / seconds) if seconds is not None and seconds > 0.0 else None
        ),
        "inference_seconds_per_example": seconds_per_unit(seconds, examples),
    }
    if steps is not None:
        summary.update(
            {
                "inference_seconds_per_step": seconds_per_unit(seconds, steps),
                "inference_seconds_per_example_per_step": seconds_per_unit(
                    seconds,
                    examples * steps,
                ),
            }
        )
    return summary
