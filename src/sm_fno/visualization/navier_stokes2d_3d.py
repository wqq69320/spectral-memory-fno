"""3D-style visualization helpers for 2D Navier-Stokes vorticity predictions."""

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
from matplotlib import animation, cm  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

_DARK_FIGURE = "#101216"
_DARK_AXIS = "#171a20"
_DARK_TEXT = "#f8fafc"
_DARK_MUTED = "#cbd5e1"


def coerce_vorticity_sequence(values: np.ndarray) -> np.ndarray:
    """Return a vorticity sequence with shape ``(time, height, width)``."""
    sequence = np.asarray(values)
    if sequence.ndim == 4 and sequence.shape[-1] == 1:
        sequence = sequence[..., 0]
    if sequence.ndim == 2:
        sequence = sequence[None, ...]
    if sequence.ndim != 3:
        raise ValueError(
            "Expected vorticity with shape (time, height, width, 1), "
            "(time, height, width), or (height, width)."
        )
    return sequence.astype(np.float32, copy=False)


def frame_indices(num_frames: int, max_frames: int | None = None) -> list[int]:
    """Return evenly spaced frame indices for an animation."""
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1.")
    if max_frames is None or max_frames >= num_frames:
        return list(range(num_frames))
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")
    return sorted(set(int(round(value)) for value in np.linspace(0, num_frames - 1, max_frames)))


def vorticity_to_velocity(
    vorticity: np.ndarray,
    *,
    domain_length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive a periodic incompressible 2D velocity field from scalar vorticity.

    This diagnostic follows the same streamfunction convention as the local
    Navier-Stokes2D generator. It is intended for visual stream/trace overlays,
    not for validating solver fidelity.
    """
    omega = np.asarray(vorticity, dtype=np.float32)
    if omega.ndim != 2:
        raise ValueError("vorticity_to_velocity expects one 2D field.")

    height, width = omega.shape
    dx = domain_length / width
    dy = domain_length / height
    k_x = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)[None, :]
    k_y = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)[:, None]
    wave_number_squared = k_x * k_x + k_y * k_y
    safe_wave_number_squared = wave_number_squared.copy()
    safe_wave_number_squared[0, 0] = 1.0

    omega_hat = np.fft.fft2(omega)
    stream_hat = omega_hat / safe_wave_number_squared
    stream_hat[0, 0] = 0.0
    velocity_x = np.fft.ifft2(1j * k_y * stream_hat).real
    velocity_y = np.fft.ifft2(-1j * k_x * stream_hat).real
    return velocity_x.astype(np.float32), velocity_y.astype(np.float32)


def _surface_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized plotting coordinates for one grid."""
    x = np.linspace(0.0, 1.0, width, endpoint=False)
    y = np.linspace(0.0, 1.0, height, endpoint=False)
    return np.meshgrid(x, y)


def _style_surface_axis(ax: plt.Axes, *, zlim: tuple[float, float], title: str) -> None:
    """Apply common 3D axis styling."""
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("vorticity")
    ax.set_zlim(*zlim)
    ax.view_init(elev=32, azim=-55)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.tick_params(labelsize=7)


def _plot_surface_panel(
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    *,
    norm: Normalize,
    cmap_name: str,
    zlim: tuple[float, float],
    title: str,
) -> None:
    """Draw one 3D vorticity surface panel."""
    cmap = cm.get_cmap(cmap_name)
    ax.plot_surface(
        x_grid,
        y_grid,
        values,
        rstride=1,
        cstride=1,
        facecolors=cmap(norm(values)),
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    _style_surface_axis(ax, zlim=zlim, title=title)


def _style_dark_surface_axis(
    ax: plt.Axes,
    *,
    zlim: tuple[float, float],
    title: str,
) -> None:
    """Apply dark presentation styling to a 3D surface axis."""
    ax.set_title(title, color=_DARK_TEXT, pad=8)
    ax.set_xlabel("x", color=_DARK_MUTED, labelpad=4)
    ax.set_ylabel("y", color=_DARK_MUTED, labelpad=4)
    ax.set_zlabel("vorticity", color=_DARK_MUTED, labelpad=4)
    ax.set_zlim(*zlim)
    ax.view_init(elev=32, azim=-55)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.tick_params(colors=_DARK_MUTED, labelsize=6, pad=0)
    ax.set_facecolor(_DARK_AXIS)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(_DARK_AXIS)
        axis.pane.set_edgecolor("#334155")
        axis._axinfo["grid"]["color"] = "#334155"


def _plot_dark_surface_panel(
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    *,
    norm: Normalize,
    zlim: tuple[float, float],
    title: str,
    cmap_name: str = "coolwarm",
) -> None:
    """Draw one dark-theme 3D-style vorticity surface panel."""
    cmap = cm.get_cmap(cmap_name)
    ax.plot_surface(
        x_grid,
        y_grid,
        values,
        rstride=1,
        cstride=1,
        facecolors=cmap(norm(values)),
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    _style_dark_surface_axis(ax, zlim=zlim, title=title)


def _resolve_animation_writer(output_path: Path, fps: int) -> animation.AbstractMovieWriter:
    """Return a Matplotlib writer for GIF or MP4 output."""
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("MP4 export requires an available ffmpeg writer.")
        return animation.FFMpegWriter(fps=fps, codec="h264")
    raise ValueError(f"Unsupported animation extension: {output_path.suffix}")


def save_vorticity_surface_rollout_animation(
    sequence: np.ndarray,
    output_path: Path,
    *,
    max_frames: int | None = 24,
    fps: int = 6,
    dpi: int = 90,
    title: str = "2D vorticity rollout shown as a 3D-style surface",
    panel_title: str = "vorticity",
    field_abs: float | None = None,
) -> None:
    """Save one dark-theme 3D-style surface animation for a 2D vorticity rollout.

    This is a 3D-style rendering of a 2D scalar vorticity field. It does not
    represent true 3D fluid forecasting.
    """
    vorticity_sequence = coerce_vorticity_sequence(sequence)
    indices = frame_indices(vorticity_sequence.shape[0], max_frames)
    if field_abs is None:
        field_abs = float(max(np.abs(vorticity_sequence).max(), 1e-6))
    norm = Normalize(vmin=-field_abs, vmax=field_abs)
    zlim = (-field_abs, field_abs)
    x_grid, y_grid = _surface_grid(vorticity_sequence.shape[1], vorticity_sequence.shape[2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.8, 4.6), facecolor=_DARK_FIGURE)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    def draw_frame(frame_number: int) -> Sequence[plt.Artist]:
        frame_index = indices[frame_number]
        ax.clear()
        _plot_dark_surface_panel(
            ax,
            x_grid,
            y_grid,
            vorticity_sequence[frame_index],
            norm=norm,
            zlim=zlim,
            title=panel_title,
        )
        fig.suptitle(
            f"{title} - rollout step {frame_index + 1}",
            color=_DARK_TEXT,
            fontsize=11,
        )
        return (ax,)

    draw_frame(0)
    surface_animation = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(indices),
        interval=1000 / max(1, fps),
        blit=False,
    )
    surface_animation.save(output_path, writer=_resolve_animation_writer(output_path, fps), dpi=dpi)
    plt.close(fig)


def save_vorticity_surface_comparison_animation(
    sequences: dict[str, np.ndarray],
    output_path: Path,
    *,
    max_frames: int | None = 24,
    fps: int = 6,
    dpi: int = 90,
    title: str = "2D vorticity rollout comparison as 3D-style surfaces",
    field_abs: float | None = None,
) -> None:
    """Save a dark-theme side-by-side 3D-style rollout comparison animation.

    Each panel is a 3D-style rendering of a 2D scalar vorticity field. The
    animation is intended for presentation diagnostics, not true 3D simulation.
    """
    if not sequences:
        raise ValueError("At least one sequence is required.")
    coerced = {label: coerce_vorticity_sequence(values) for label, values in sequences.items()}
    shapes = {values.shape for values in coerced.values()}
    if len(shapes) != 1:
        raise ValueError("All comparison sequences must have matching shapes.")

    first_sequence = next(iter(coerced.values()))
    indices = frame_indices(first_sequence.shape[0], max_frames)
    if field_abs is None:
        field_abs = float(max(np.abs(values).max() for values in coerced.values()))
        field_abs = max(field_abs, 1e-6)
    norm = Normalize(vmin=-field_abs, vmax=field_abs)
    zlim = (-field_abs, field_abs)
    x_grid, y_grid = _surface_grid(first_sequence.shape[1], first_sequence.shape[2])
    labels = list(coerced)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(3.8 * len(labels), 4.2), facecolor=_DARK_FIGURE)
    axes = [
        fig.add_subplot(1, len(labels), index + 1, projection="3d")
        for index in range(len(labels))
    ]

    def draw_frame(frame_number: int) -> Sequence[plt.Artist]:
        frame_index = indices[frame_number]
        for ax, label in zip(axes, labels, strict=True):
            ax.clear()
            _plot_dark_surface_panel(
                ax,
                x_grid,
                y_grid,
                coerced[label][frame_index],
                norm=norm,
                zlim=zlim,
                title=label,
            )
        fig.suptitle(
            f"{title} - rollout step {frame_index + 1}",
            color=_DARK_TEXT,
            fontsize=11,
        )
        return axes

    draw_frame(0)
    comparison_animation = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(indices),
        interval=1000 / max(1, fps),
        blit=False,
    )
    comparison_animation.save(
        output_path,
        writer=_resolve_animation_writer(output_path, fps),
        dpi=dpi,
    )
    plt.close(fig)


def plot_vorticity_surface_triptych(
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    *,
    title: str = "2D vorticity shown as 3D-style surfaces",
    dpi: int = 180,
) -> None:
    """Plot true, predicted, and absolute-error 2D vorticity as surfaces."""
    target_field = np.asarray(target, dtype=np.float32)
    prediction_field = np.asarray(prediction, dtype=np.float32)
    if target_field.shape != prediction_field.shape:
        raise ValueError("target and prediction fields must have the same shape.")
    if target_field.ndim != 2:
        raise ValueError("surface triptych expects 2D fields.")

    error_field = np.abs(prediction_field - target_field)
    x_grid, y_grid = _surface_grid(*target_field.shape)
    field_abs = float(max(np.abs(target_field).max(), np.abs(prediction_field).max(), 1e-6))
    error_max = float(max(error_field.max(), 1e-6))
    field_norm = Normalize(vmin=-field_abs, vmax=field_abs)
    error_norm = Normalize(vmin=0.0, vmax=error_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 4.6))
    axes = [fig.add_subplot(1, 3, index, projection="3d") for index in range(1, 4)]
    _plot_surface_panel(
        axes[0],
        x_grid,
        y_grid,
        target_field,
        norm=field_norm,
        cmap_name="coolwarm",
        zlim=(-field_abs, field_abs),
        title="true vorticity",
    )
    _plot_surface_panel(
        axes[1],
        x_grid,
        y_grid,
        prediction_field,
        norm=field_norm,
        cmap_name="coolwarm",
        zlim=(-field_abs, field_abs),
        title="predicted vorticity",
    )
    _plot_surface_panel(
        axes[2],
        x_grid,
        y_grid,
        error_field,
        norm=error_norm,
        cmap_name="magma",
        zlim=(0.0, error_max),
        title="absolute error",
    )

    fig.suptitle(title)
    field_mappable = cm.ScalarMappable(norm=field_norm, cmap="coolwarm")
    error_mappable = cm.ScalarMappable(norm=error_norm, cmap="magma")
    fig.colorbar(field_mappable, ax=axes[:2], shrink=0.62, pad=0.02, label="vorticity")
    fig.colorbar(error_mappable, ax=axes[2], shrink=0.62, pad=0.08, label="absolute error")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_rollout_surface_animation(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    *,
    max_frames: int | None = 36,
    fps: int = 6,
    dpi: int = 120,
    title: str = "2D vorticity rollout shown as 3D-style surfaces",
) -> None:
    """Save a GIF animation of true, predicted, and error vorticity surfaces."""
    target_sequence = coerce_vorticity_sequence(targets)
    prediction_sequence = coerce_vorticity_sequence(predictions)
    if target_sequence.shape != prediction_sequence.shape:
        raise ValueError("target and prediction sequences must have the same shape.")

    indices = frame_indices(target_sequence.shape[0], max_frames)
    field_abs = float(
        max(np.abs(target_sequence).max(), np.abs(prediction_sequence).max(), 1e-6)
    )
    error_max = float(max(np.abs(prediction_sequence - target_sequence).max(), 1e-6))
    field_norm = Normalize(vmin=-field_abs, vmax=field_abs)
    error_norm = Normalize(vmin=0.0, vmax=error_max)
    x_grid, y_grid = _surface_grid(target_sequence.shape[1], target_sequence.shape[2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 4.6))
    axes = [fig.add_subplot(1, 3, index, projection="3d") for index in range(1, 4)]

    def draw_frame(frame_number: int) -> Sequence[plt.Artist]:
        frame_index = indices[frame_number]
        target_field = target_sequence[frame_index]
        prediction_field = prediction_sequence[frame_index]
        error_field = np.abs(prediction_field - target_field)
        for ax in axes:
            ax.clear()
        _plot_surface_panel(
            axes[0],
            x_grid,
            y_grid,
            target_field,
            norm=field_norm,
            cmap_name="coolwarm",
            zlim=(-field_abs, field_abs),
            title="true vorticity",
        )
        _plot_surface_panel(
            axes[1],
            x_grid,
            y_grid,
            prediction_field,
            norm=field_norm,
            cmap_name="coolwarm",
            zlim=(-field_abs, field_abs),
            title="predicted vorticity",
        )
        _plot_surface_panel(
            axes[2],
            x_grid,
            y_grid,
            error_field,
            norm=error_norm,
            cmap_name="magma",
            zlim=(0.0, error_max),
            title="absolute error",
        )
        fig.suptitle(f"{title} - rollout step {frame_index + 1}")
        return axes

    draw_frame(0)
    surface_animation = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(indices),
        interval=1000 / max(1, fps),
        blit=False,
    )
    writer = animation.PillowWriter(fps=fps)
    surface_animation.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)


def _sample_periodic(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinearly sample a periodic field at normalized coordinates."""
    height, width = field.shape
    x_position = (x % 1.0) * width
    y_position = (y % 1.0) * height
    x0 = np.floor(x_position).astype(int) % width
    y0 = np.floor(y_position).astype(int) % height
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    wx = x_position - np.floor(x_position)
    wy = y_position - np.floor(y_position)
    return (
        (1.0 - wx) * (1.0 - wy) * field[y0, x0]
        + wx * (1.0 - wy) * field[y0, x1]
        + (1.0 - wx) * wy * field[y1, x0]
        + wx * wy * field[y1, x1]
    )


def integrate_particle_traces(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    *,
    particles_per_axis: int = 8,
    steps: int = 45,
) -> np.ndarray:
    """Integrate short periodic traces through one stationary velocity field."""
    if velocity_x.shape != velocity_y.shape:
        raise ValueError("velocity components must have matching shapes.")
    if particles_per_axis < 1:
        raise ValueError("particles_per_axis must be at least 1.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    seed_axis = np.linspace(0.08, 0.92, particles_per_axis)
    seed_x, seed_y = np.meshgrid(seed_axis, seed_axis)
    x = seed_x.reshape(-1)
    y = seed_y.reshape(-1)
    traces = np.empty((x.size, steps + 1, 2), dtype=np.float32)
    traces[:, 0, 0] = x
    traces[:, 0, 1] = y

    max_speed = float(np.sqrt(velocity_x * velocity_x + velocity_y * velocity_y).max())
    dt = 0.35 / (max(max_speed, 1e-6) * steps)
    for step in range(1, steps + 1):
        vx = _sample_periodic(velocity_x, x, y)
        vy = _sample_periodic(velocity_y, x, y)
        x = (x + dt * vx) % 1.0
        y = (y + dt * vy) % 1.0
        traces[:, step, 0] = x
        traces[:, step, 1] = y
    return traces


def plot_velocity_trace_diagnostic(
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    *,
    particles_per_axis: int = 8,
    trace_steps: int = 45,
    dpi: int = 180,
) -> None:
    """Plot a simple vorticity-derived velocity trace diagnostic."""
    target_field = np.asarray(target, dtype=np.float32)
    prediction_field = np.asarray(prediction, dtype=np.float32)
    if target_field.shape != prediction_field.shape:
        raise ValueError("target and prediction fields must have the same shape.")
    if target_field.ndim != 2:
        raise ValueError("velocity trace diagnostic expects 2D fields.")

    target_velocity = vorticity_to_velocity(target_field)
    prediction_velocity = vorticity_to_velocity(prediction_field)
    target_traces = integrate_particle_traces(
        *target_velocity,
        particles_per_axis=particles_per_axis,
        steps=trace_steps,
    )
    prediction_traces = integrate_particle_traces(
        *prediction_velocity,
        particles_per_axis=particles_per_axis,
        steps=trace_steps,
    )
    error_field = np.abs(prediction_field - target_field)
    field_abs = float(max(np.abs(target_field).max(), np.abs(prediction_field).max(), 1e-6))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = (
        ("true vorticity-derived traces", target_field, target_traces),
        ("predicted vorticity-derived traces", prediction_field, prediction_traces),
        ("absolute vorticity error", error_field, None),
    )
    for ax, (panel_title, field, traces) in zip(axes, panels, strict=True):
        if traces is None:
            image = ax.imshow(field, origin="lower", cmap="magma")
        else:
            image = ax.imshow(
                field,
                origin="lower",
                cmap="coolwarm",
                vmin=-field_abs,
                vmax=field_abs,
            )
            height, width = field.shape
            for trace in traces:
                ax.plot(
                    trace[:, 0] * (width - 1),
                    trace[:, 1] * (height - 1),
                    color="black",
                    alpha=0.55,
                    linewidth=0.6,
                )
                ax.scatter(
                    trace[0, 0] * (width - 1),
                    trace[0, 1] * (height - 1),
                    color="white",
                    edgecolor="black",
                    linewidth=0.25,
                    s=7,
                    zorder=3,
                )
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("2D incompressible velocity traces derived from vorticity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
