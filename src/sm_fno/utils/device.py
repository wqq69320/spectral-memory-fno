"""Device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device string into a PyTorch device."""
    normalized = preference.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized in {"cpu", "cuda", "mps"}:
        return torch.device(normalized)
    raise ValueError(f"Unsupported device preference: {preference}")
