"""Path helpers that avoid hardcoded absolute locations."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root inferred from this file location."""
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
