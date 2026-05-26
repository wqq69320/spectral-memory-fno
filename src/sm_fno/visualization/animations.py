"""Animation helpers for PDE fields."""

from __future__ import annotations

from pathlib import Path

import torch


def save_rollout_animation(fields: torch.Tensor, output_path: Path) -> None:
    """Save a rollout animation.

    TODO: Implement Matplotlib animation export once field conventions are
    finalized.
    """
    del fields, output_path
    raise NotImplementedError("Animation export is not implemented in the scaffold phase.")
