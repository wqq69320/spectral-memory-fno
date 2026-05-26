"""Optimizer builders."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build an optimizer from a small supported set."""
    normalized = name.lower()
    if normalized == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")
