"""Plotting helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

_CACHE_ROOT = Path(tempfile.gettempdir()) / "sm_fno_plot_cache"
_MPL_CONFIG_DIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_DIR = _CACHE_ROOT / "xdg"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_rollout_error(errors: Sequence[float], output_path: Path | None = None) -> None:
    """Plot rollout error over forecast steps."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(errors) + 1), errors, marker="o")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Error")
    ax.set_title("Long-horizon rollout error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_training_loss(history: dict[str, Sequence[float]], output_path: Path) -> None:
    """Plot training and optional validation loss curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    train_loss = history.get("train_loss", [])
    if train_loss:
        ax.plot(range(1, len(train_loss) + 1), train_loss, marker="o", label="train")
    val_loss = history.get("val_loss", [])
    if val_loss:
        ax.plot(range(1, len(val_loss) + 1), val_loss, marker="s", label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    if train_loss or val_loss:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_heat1d_prediction(
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    *,
    input_state: np.ndarray | None = None,
    step_index: int = -1,
) -> None:
    """Plot one true-vs-predicted 1D or 2D PDE forecast state."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_state = np.asarray(target)[step_index, ..., 0]
    prediction_state = np.asarray(prediction)[step_index, ..., 0]
    if target_state.ndim == 1:
        x = np.arange(target_state.shape[0])
        fig, ax = plt.subplots(figsize=(7, 4))
        if input_state is not None:
            ax.plot(
                x,
                np.asarray(input_state)[-1, :, 0],
                color="0.6",
                linestyle="--",
                label="input last",
            )
        ax.plot(x, target_state, label="true")
        ax.plot(x, prediction_state, label="predicted")
        ax.set_xlabel("Grid index")
        ax.set_ylabel("u")
        ax.set_title("1D PDE prediction")
        ax.grid(True, alpha=0.3)
        ax.legend()
    elif target_state.ndim == 2:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        panels = (
            ("true", target_state),
            ("predicted", prediction_state),
            ("absolute error", np.abs(prediction_state - target_state)),
        )
        for ax, (title, values) in zip(axes, panels, strict=True):
            image = ax.imshow(values, origin="lower", cmap="viridis")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("2D PDE prediction")
    else:
        raise ValueError("prediction plotting expects 1D or 2D spatial fields.")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
