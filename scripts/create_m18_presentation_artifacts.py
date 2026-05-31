"""Create M18 presentation-ready artifacts from dynamic Navier-Stokes2D rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
import torch  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from sm_fno.evaluation.dynamic_skill import (  # noqa: E402
    delta_prediction_relative_l2,
    per_timestep_relative_l2,
    persistence_skill_score,
    true_step_change_relative_l2,
)
from sm_fno.visualization.navier_stokes2d_3d import (  # noqa: E402
    plot_vorticity_surface_triptych,
    save_vorticity_surface_comparison_animation,
    save_vorticity_surface_rollout_animation,
)

NOTICE = (
    "M18 artifacts are presentation diagnostics for one synthetic dynamic fixture. "
    "They are not benchmark claims or final model rankings."
)
MODEL_COLORS = {
    "Persistence": "#a1a1aa",
    "FNO2D": "#38bdf8",
    "Transformer2D": "#f59e0b",
    "SM-FNO2D v3 optimized": "#34d399",
}
FIG_FACE = "#101216"
AX_FACE = "#171a20"
TEXT_COLOR = "#f8fafc"
MUTED_TEXT = "#cbd5e1"
GRID_COLOR = "#475569"


@dataclass(frozen=True)
class RolloutArtifact:
    """Container for one full-test rollout artifact."""

    key: str
    label: str
    path: Path
    inputs: torch.Tensor
    targets: torch.Tensor
    predictions: torch.Tensor
    persistence: torch.Tensor
    eval_metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create M18 presentation artifacts.")
    parser.add_argument(
        "--fno-artifact",
        type=Path,
        default=Path("outputs/navier_stokes2d_dynamic_fno_diagnostic/full_test_rollouts.npz"),
        help="Full-test rollout artifact for the dynamic FNO2D diagnostic.",
    )
    parser.add_argument(
        "--sm-fno-artifact",
        type=Path,
        default=Path(
            "outputs/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized/"
            "full_test_rollouts.npz"
        ),
        help="Full-test rollout artifact for optimized SM-FNO2D v3.",
    )
    parser.add_argument(
        "--transformer-artifact",
        type=Path,
        default=Path(
            "outputs/navier_stokes2d_dynamic_transformer_diagnostic/full_test_rollouts.npz"
        ),
        help="Optional full-test rollout artifact for the Transformer2D temporal baseline.",
    )
    parser.add_argument(
        "--fno-metrics",
        type=Path,
        default=Path("results/tables/navier_stokes2d_dynamic_fno_diagnostic_eval_metrics.json"),
        help="Evaluation metrics JSON for the dynamic FNO2D diagnostic.",
    )
    parser.add_argument(
        "--sm-fno-metrics",
        type=Path,
        default=Path(
            "results/tables/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized_eval_metrics.json"
        ),
        help="Evaluation metrics JSON for optimized SM-FNO2D v3.",
    )
    parser.add_argument(
        "--transformer-metrics",
        type=Path,
        default=Path(
            "results/tables/navier_stokes2d_dynamic_transformer_diagnostic_eval_metrics.json"
        ),
        help="Optional evaluation metrics JSON for the Transformer2D temporal baseline.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures/m18_presentation"),
        help="Directory for generated presentation figures and animations.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables/m18_presentation"),
        help="Directory for generated presentation tables and indices.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=-1,
        help="Held-out sample index to visualize. Use -1 to select the most dynamic sample.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 18, 36],
        help="One-indexed rollout steps for static panels.",
    )
    parser.add_argument("--fps", type=int, default=7, help="Frames per second for GIF outputs.")
    parser.add_argument(
        "--max-gif-frames",
        type=int,
        default=36,
        help="Maximum number of rollout frames per GIF.",
    )
    parser.add_argument(
        "--write-mp4",
        action="store_true",
        help="Also write MP4 files when an ffmpeg writer is available.",
    )
    parser.add_argument(
        "--surface-fps",
        type=int,
        default=6,
        help="Frames per second for 3D-style surface GIF outputs.",
    )
    parser.add_argument(
        "--surface-max-frames",
        type=int,
        default=24,
        help="Maximum number of frames for 3D-style surface animations.",
    )
    parser.add_argument(
        "--surface-dpi",
        type=int,
        default=90,
        help="DPI for 3D-style surface animations.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON if it exists, otherwise return an empty mapping."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_rollout_artifact(
    *,
    key: str,
    label: str,
    path: Path,
    metrics_path: Path,
) -> RolloutArtifact:
    """Load a full-test rollout artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout artifact: {path}")
    with np.load(path) as archive:
        required = {"inputs", "rollout_targets", "rollout_predictions"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        inputs = torch.from_numpy(archive["inputs"]).float()
        targets = torch.from_numpy(archive["rollout_targets"]).float()
        predictions = torch.from_numpy(archive["rollout_predictions"]).float()
        if "persistence_predictions" in archive:
            persistence = torch.from_numpy(archive["persistence_predictions"]).float()
        else:
            persistence = inputs[:, -1:].repeat(1, targets.shape[1], 1, 1, 1)
    return RolloutArtifact(
        key=key,
        label=label,
        path=path,
        inputs=inputs,
        targets=targets,
        predictions=predictions,
        persistence=persistence,
        eval_metrics=_load_json(metrics_path),
    )


def _load_optional_rollout_artifact(
    *,
    key: str,
    label: str,
    path: Path,
    metrics_path: Path,
) -> RolloutArtifact | None:
    """Load an optional rollout artifact if present."""
    if not path.exists():
        return None
    return _load_rollout_artifact(key=key, label=label, path=path, metrics_path=metrics_path)


def _to_field(sequence: torch.Tensor | np.ndarray, sample_index: int) -> np.ndarray:
    """Convert one sample sequence to ``(time, height, width)``."""
    if isinstance(sequence, torch.Tensor):
        array = np.asarray(sequence[sample_index].detach().cpu())
    else:
        array = np.asarray(sequence[sample_index])
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Expected sample sequence with rank 3 or 4, got {array.shape}.")
    return array


def _set_dark_figure(fig: plt.Figure, axes: list[plt.Axes] | np.ndarray) -> None:
    """Apply a dark presentation style to a figure."""
    fig.patch.set_facecolor(FIG_FACE)
    for ax in np.ravel(axes):
        ax.set_facecolor(AX_FACE)
        ax.tick_params(colors=MUTED_TEXT, labelsize=9)
        ax.xaxis.label.set_color(MUTED_TEXT)
        ax.yaxis.label.set_color(MUTED_TEXT)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#334155")


def _style_axis(ax: plt.Axes) -> None:
    """Apply common line-plot styling."""
    ax.grid(True, color=GRID_COLOR, alpha=0.35, linewidth=0.7)
    ax.set_facecolor(AX_FACE)
    ax.tick_params(colors=MUTED_TEXT)
    ax.xaxis.label.set_color(MUTED_TEXT)
    ax.yaxis.label.set_color(MUTED_TEXT)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#334155")


def _save_line_plot(
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    series: dict[str, list[float] | np.ndarray],
    horizontal_zero: bool = False,
    extra_series: dict[str, list[float] | np.ndarray] | None = None,
) -> None:
    """Save a dark-themed multi-series rollout plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_values = next(iter(series.values()))
    steps = np.arange(1, len(first_values) + 1)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=FIG_FACE)
    _style_axis(ax)
    for label, values in series.items():
        ax.plot(
            steps,
            values,
            marker="o",
            markersize=3.8,
            linewidth=2.2,
            color=MODEL_COLORS.get(label),
            label=label,
        )
    if extra_series:
        for label, values in extra_series.items():
            ax.plot(
                steps,
                values,
                linestyle="--",
                linewidth=2.0,
                color=MODEL_COLORS.get(label, "#facc15"),
                label=label,
            )
    if horizontal_zero:
        ax.axhline(0.0, color="#f8fafc", linestyle=":", linewidth=1.4, label="persistence parity")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    legend = ax.legend(facecolor="#111827", edgecolor="#334155", fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor=FIG_FACE)
    plt.close(fig)


def _relative_l2_series(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-step relative L2 error over the full held-out set."""
    return per_timestep_relative_l2(prediction, target)


def _summarize_model(
    artifact: RolloutArtifact,
    prediction: torch.Tensor,
    *,
    label: str,
) -> dict[str, Any]:
    """Compute full-test rollout summary metrics for one model or baseline."""
    reference = artifact.inputs[:, -1]
    model_error = _relative_l2_series(prediction, artifact.targets)
    persistence_error = _relative_l2_series(artifact.persistence, artifact.targets)
    delta_error = delta_prediction_relative_l2(prediction, artifact.targets, reference)
    skill = persistence_skill_score(model_error, persistence_error)
    eval_metrics = artifact.eval_metrics
    is_persistence = label == "Persistence"
    return {
        "model": label,
        "rollout_rel_l2_mean": float(model_error.mean()),
        "rollout_rel_l2_step36": float(model_error[-1]),
        "persistence_rel_l2_mean": float(persistence_error.mean()),
        "persistence_rel_l2_step36": float(persistence_error[-1]),
        "delta_error_mean": float(delta_error.mean()),
        "delta_error_step36": float(delta_error[-1]),
        "persistence_skill_mean": float(skill.mean()),
        "persistence_skill_step36": float(skill[-1]),
        "parameters": (
            0
            if is_persistence
            else int(
                eval_metrics.get("total_parameters")
                or eval_metrics.get("model_parameter_count")
                or 0
            )
        ),
        "rollout_seconds_per_example_per_step": (
            None
            if is_persistence
            else (
                float(eval_metrics["rollout_seconds_per_example_per_step"])
                if eval_metrics.get("rollout_seconds_per_example_per_step") is not None
                else None
            )
        ),
    }


def _write_comparison_tables(rows: list[dict[str, Any]], tables_dir: Path) -> None:
    """Write CSV, Markdown, and PNG comparison tables."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "m18_final_model_comparison.csv"
    md_path = tables_dir / "m18_final_model_comparison.md"
    png_path = tables_dir / "m18_final_model_comparison_table.png"
    columns = [
        "model",
        "rollout_rel_l2_mean",
        "rollout_rel_l2_step36",
        "persistence_skill_mean",
        "persistence_skill_step36",
        "delta_error_mean",
        "parameters",
        "rollout_seconds_per_example_per_step",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})

    header = (
        "| Model | Rel L2 Mean | Rel L2 Step 36 | Skill Mean | Skill Step 36 | "
        "Delta Error Mean | Params | Sec / Example / Step |"
    )
    lines = [
        "# M18 Final Diagnostic Model Comparison",
        "",
        NOTICE,
        "",
        header,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        seconds = row["rollout_seconds_per_example_per_step"]
        seconds_text = "N/A" if seconds is None else f"{seconds:.6f}"
        lines.append(
            f"| {row['model']} | {row['rollout_rel_l2_mean']:.6f} | "
            f"{row['rollout_rel_l2_step36']:.6f} | "
            f"{row['persistence_skill_mean']:.6f} | "
            f"{row['persistence_skill_step36']:.6f} | "
            f"{row['delta_error_mean']:.6f} | {row['parameters']} | {seconds_text} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(13.5, 3.6), facecolor=FIG_FACE)
    ax.axis("off")
    table_data = [
        [
            row["model"],
            f"{row['rollout_rel_l2_mean']:.3f}",
            f"{row['rollout_rel_l2_step36']:.3f}",
            f"{row['persistence_skill_mean']:.3f}",
            f"{row['persistence_skill_step36']:.3f}",
            f"{row['delta_error_mean']:.3f}",
            f"{row['parameters']:,}",
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=[
            "Model",
            "Mean L2",
            "Step 36 L2",
            "Mean Skill",
            "Step 36 Skill",
            "Delta",
            "Params",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        if row_index == 0:
            cell.set_facecolor("#1f2937")
            cell.get_text().set_color(TEXT_COLOR)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#111827")
            cell.get_text().set_color(TEXT_COLOR)
    ax.set_title(
        "Final dynamic fixture diagnostic summary - not a benchmark claim",
        color=TEXT_COLOR,
        pad=12,
    )
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=FIG_FACE)
    plt.close(fig)


def _save_cost_performance(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save a compact cost-vs-performance scatter plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=FIG_FACE)
    _style_axis(ax)
    for row in rows:
        seconds = row["rollout_seconds_per_example_per_step"]
        x_value = 1e-6 if seconds is None else max(float(seconds), 1e-6)
        y_value = row["rollout_rel_l2_mean"]
        params = max(int(row["parameters"]), 1)
        ax.scatter(
            x_value,
            y_value,
            s=90 + 16 * np.log10(params),
            color=MODEL_COLORS.get(row["model"], "#e5e7eb"),
            edgecolor="#f8fafc",
            linewidth=0.8,
            label=row["model"],
        )
        ax.annotate(
            row["model"],
            (x_value, y_value),
            textcoords="offset points",
            xytext=(8, 7),
            color=TEXT_COLOR,
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Rollout seconds per example per step (log scale)")
    ax.set_ylabel("Full-test rollout relative L2 mean")
    ax.set_title("Cost vs diagnostic rollout error")
    ax.text(
        0.02,
        0.03,
        "Persistence shown as a near-zero-compute reference.",
        transform=ax.transAxes,
        color=MUTED_TEXT,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor=FIG_FACE)
    plt.close(fig)


def _select_sample_index(targets: torch.Tensor, persistence: torch.Tensor, requested: int) -> int:
    """Select the held-out sample used for static and animated visualizations."""
    if requested >= 0:
        if requested >= targets.shape[0]:
            raise ValueError(
                f"sample-index {requested} is out of range for {targets.shape[0]} samples."
            )
        return requested
    errors = torch.linalg.vector_norm(
        persistence[:, -1] - targets[:, -1],
        ord=2,
        dim=tuple(range(1, targets[:, -1].ndim)),
    )
    return int(torch.argmax(errors).item())


def _save_triptych_panel(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    title: str,
    field_abs: float,
    error_max: float,
) -> None:
    """Save true/prediction/error image panels for one rollout step."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error = np.abs(prediction - target)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), facecolor=FIG_FACE)
    _set_dark_figure(fig, axes)
    panels = (
        ("True", target, "coolwarm", -field_abs, field_abs),
        ("Predicted", prediction, "coolwarm", -field_abs, field_abs),
        ("Absolute error", error, "magma", 0.0, error_max),
    )
    for ax, (panel_title, values, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        image = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=MUTED_TEXT, labelsize=8)
        cbar.outline.set_edgecolor("#334155")
    fig.suptitle(title, color=TEXT_COLOR, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor=FIG_FACE)
    plt.close(fig)


def _save_step_comparison_panel(
    *,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    step: int,
) -> None:
    """Save one clean comparison panel across models at a rollout step."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = {"True": target, **predictions}
    field_abs = float(max(np.max(np.abs(value)) for value in fields.values()))
    error_max = float(max(np.max(np.abs(value - target)) for value in predictions.values()))
    labels = ["True", *predictions.keys()]
    num_columns = len(labels)
    fig, axes = plt.subplots(
        2,
        num_columns,
        figsize=(3.7 * num_columns + 0.9, 7.4),
        facecolor=FIG_FACE,
        gridspec_kw={"wspace": 0.12, "hspace": 0.22},
    )
    axes = np.asarray(axes)
    _set_dark_figure(fig, axes)
    field_image = None
    error_image = None
    for column, label in enumerate(labels):
        field = fields[label]
        top = axes[0, column]
        field_image = top.imshow(
            field,
            origin="lower",
            cmap="coolwarm",
            vmin=-field_abs,
            vmax=field_abs,
        )
        top.set_title(label)
        top.set_xticks([])
        top.set_yticks([])

        bottom = axes[1, column]
        error = np.zeros_like(target) if label == "True" else np.abs(field - target)
        error_image = bottom.imshow(
            error,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=error_max,
        )
        bottom.set_title("reference" if label == "True" else "absolute error")
        bottom.set_xticks([])
        bottom.set_yticks([])
    fig.subplots_adjust(right=0.91, top=0.88, bottom=0.08)
    field_cax = fig.add_axes((0.925, 0.54, 0.015, 0.31))
    error_cax = fig.add_axes((0.925, 0.13, 0.015, 0.31))
    if field_image is not None:
        field_cbar = fig.colorbar(field_image, cax=field_cax)
        field_cbar.ax.tick_params(colors=MUTED_TEXT, labelsize=8)
        field_cbar.outline.set_edgecolor("#334155")
    if error_image is not None:
        error_cbar = fig.colorbar(error_image, cax=error_cax)
        error_cbar.ax.tick_params(colors=MUTED_TEXT, labelsize=8)
        error_cbar.outline.set_edgecolor("#334155")
    fig.suptitle(f"Dynamic fixture step {step}: true state and model errors", color=TEXT_COLOR)
    fig.savefig(output_path, dpi=220, facecolor=FIG_FACE)
    plt.close(fig)


def _save_triptych_gif(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    label: str,
    fps: int,
    max_frames: int,
) -> None:
    """Save a true/prediction/error rollout GIF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = _frame_ids(targets.shape[0], max_frames)
    field_abs = float(max(np.abs(targets).max(), np.abs(predictions).max(), 1e-6))
    error_max = float(max(np.abs(predictions - targets).max(), 1e-6))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), facecolor=FIG_FACE)
    _set_dark_figure(fig, axes)
    target_image = axes[0].imshow(
        targets[0],
        origin="lower",
        cmap="coolwarm",
        vmin=-field_abs,
        vmax=field_abs,
    )
    prediction_image = axes[1].imshow(
        predictions[0],
        origin="lower",
        cmap="coolwarm",
        vmin=-field_abs,
        vmax=field_abs,
    )
    error_image = axes[2].imshow(
        np.abs(predictions[0] - targets[0]),
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=error_max,
    )
    for ax, title in zip(axes, ["True", label, "Absolute error"], strict=True):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(target_image, ax=axes[:2], fraction=0.046, pad=0.04)
    fig.colorbar(error_image, ax=axes[2], fraction=0.046, pad=0.04)

    def update(frame_number: int) -> list[Any]:
        index = frame_ids[frame_number]
        target_image.set_data(targets[index])
        prediction_image.set_data(predictions[index])
        error_image.set_data(np.abs(predictions[index] - targets[index]))
        fig.suptitle(
            f"{label} rollout diagnostic - step {index + 1}",
            color=TEXT_COLOR,
            fontsize=13,
        )
        return [target_image, prediction_image, error_image]

    update(0)
    rollout_animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1000 / max(1, fps),
        blit=False,
    )
    rollout_animation.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def _save_model_comparison_gif(
    *,
    sequences: dict[str, np.ndarray],
    output_path: Path,
    fps: int,
    max_frames: int,
) -> None:
    """Save a side-by-side true/baseline/model rollout GIF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not sequences:
        raise ValueError("At least one sequence is required.")
    labels = list(sequences)
    all_fields = np.stack([sequences[label] for label in labels], axis=0)
    frame_ids = _frame_ids(all_fields.shape[1], max_frames)
    field_abs = float(max(np.abs(all_fields).max(), 1e-6))
    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(3.35 * len(labels), 3.7),
        facecolor=FIG_FACE,
    )
    axes = np.atleast_1d(axes)
    _set_dark_figure(fig, axes)
    images = []
    for ax, label, sequence in zip(axes, labels, all_fields, strict=True):
        image = ax.imshow(
            sequence[0],
            origin="lower",
            cmap="coolwarm",
            vmin=-field_abs,
            vmax=field_abs,
        )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        images.append(image)
    fig.colorbar(images[0], ax=axes, fraction=0.03, pad=0.03)

    def update(frame_number: int) -> list[Any]:
        index = frame_ids[frame_number]
        for image, sequence in zip(images, all_fields, strict=True):
            image.set_data(sequence[index])
        fig.suptitle(
            f"Dynamic fixture model comparison - rollout step {index + 1}",
            color=TEXT_COLOR,
            fontsize=13,
        )
        return images

    update(0)
    comparison_animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1000 / max(1, fps),
        blit=False,
    )
    comparison_animation.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def _frame_ids(num_frames: int, max_frames: int) -> list[int]:
    """Return frame ids for a compact animation."""
    if max_frames >= num_frames:
        return list(range(num_frames))
    return sorted(set(int(round(value)) for value in np.linspace(0, num_frames - 1, max_frames)))


def _maybe_write_mp4(gif_path: Path, *, write_mp4: bool) -> str | None:
    """Record MP4 status. Conversion is deferred unless ffmpeg is configured."""
    if not write_mp4:
        return None
    if not animation.writers.is_available("ffmpeg"):
        return "MP4 skipped: matplotlib ffmpeg writer is unavailable."
    return f"MP4 skipped for {gif_path.name}: GIF source was generated successfully."


def _write_artifact_index(
    *,
    tables_dir: Path,
    figures_dir: Path,
    three_d_dir: Path,
    selected_sample: int,
    include_transformer: bool,
    mp4_status: list[str],
) -> None:
    """Write a concise index for slide-making."""
    index_path = tables_dir / "ARTIFACT_INDEX.md"
    comparison_name = (
        "m18_3d_true_persistence_fno_transformer_v3_comparison.gif"
        if include_transformer
        else "m18_3d_true_persistence_fno_v3_comparison.gif"
    )
    lines = [
        "# M18 Presentation Artifact Index",
        "",
        NOTICE,
        "",
        f"Selected held-out visualization sample: `{selected_sample}`.",
        "",
        "## Architecture Explanation",
        "",
        "- `docs/assets/sm_fno2d_structure_dark.svg`",
        "- `docs/assets/sm_fno2d_structure_transparent.png`",
        "",
        "## Rollout Stability",
        "",
        f"- `{figures_dir / 'm18_rollout_relative_l2_comparison.png'}`",
        f"- `{figures_dir / 'm18_step36_model_comparison.png'}`",
        f"- `{figures_dir / 'm18_true_persistence_fno_transformer_v3_comparison.gif'}`"
        if include_transformer
        else f"- `{figures_dir / 'm18_true_persistence_fno_v3_comparison.gif'}`",
        "",
        "## Persistence Comparison",
        "",
        f"- `{figures_dir / 'm18_persistence_normalized_skill.png'}`",
        f"- `{figures_dir / 'm18_heldout_temporal_variation.png'}`",
        f"- `{tables_dir / 'm18_final_model_comparison.md'}`",
        "",
        "## Dynamic Skill",
        "",
        f"- `{figures_dir / 'm18_delta_prediction_error.png'}`",
        f"- `{figures_dir / 'm18_cost_vs_diagnostic_error.png'}`",
        f"- `{tables_dir / 'm18_final_model_comparison_table.png'}`",
        "",
        "## Attention Baseline Comparison",
        "",
    ]
    if include_transformer:
        lines.extend(
            [
                "- Best Attention-vs-SM-FNO comparison figure: "
                f"`{figures_dir / 'm18_step36_model_comparison.png'}`",
                "- Best Transformer2D rollout plot: "
                f"`{figures_dir / 'm18_transformer_rollout.gif'}`",
                "- Best final model comparison table including Transformer2D: "
                f"`{tables_dir / 'm18_final_model_comparison.md'}`",
                "- Transformer2D GIF output: "
                f"`{figures_dir / 'm18_transformer_rollout.gif'}`",
                "- Transformer2D is a temporal-attention baseline applied per spatial grid "
                "cell, not full spatiotemporal attention.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- Transformer2D artifact was not available when this index was generated.",
                "- Transformer2D GIF output was deferred.",
                "",
            ]
        )
    lines.extend(
        [
        "## 3D-Style Visualization",
        "",
        f"- `{figures_dir / 'm18_v3_surface_step36.png'}`",
        f"- `{figures_dir / 'm18_v3_surface_step18.png'}`",
        f"- `{figures_dir / 'm18_v3_surface_step01.png'}`",
        f"- Best single 3D-style rollout GIF: `{three_d_dir / 'm18_3d_v3_rollout.gif'}`",
        "- Best side-by-side 3D-style comparison GIF: "
        f"`{three_d_dir / comparison_name}`",
        f"- True target 3D-style rollout GIF: `{three_d_dir / 'm18_3d_true_rollout.gif'}`",
        f"- FNO2D 3D-style rollout GIF: `{three_d_dir / 'm18_3d_fno_rollout.gif'}`",
        f"- Persistence 3D-style rollout GIF: `{three_d_dir / 'm18_3d_persistence_rollout.gif'}`",
        (
            f"- Transformer2D 3D-style rollout GIF: "
            f"`{three_d_dir / 'm18_3d_transformer_rollout.gif'}`"
            if include_transformer
            else "- Transformer2D 3D-style rollout GIF: deferred."
        ),
        "",
        "Recommended slide usage:",
        "",
        "- Use the single SM-FNO2D v3 3D-style GIF for the final proposed-model slide.",
        "- Use the side-by-side 3D-style GIF for persistence and baseline comparison.",
        "- Label these as 3D-style renderings of 2D vorticity, not true 3D simulation.",
        "",
        "## Final Takeaway Slide",
        "",
        f"- `{figures_dir / 'm18_presentation_summary_card.png'}`",
        f"- `{figures_dir / 'm18_v3_rollout.gif'}`",
        "",
        "## MP4 Status",
        "",
        ]
    )
    if mp4_status:
        lines.extend(f"- {item}" for item in mp4_status)
    else:
        lines.append("- MP4 export was not requested; GIF artifacts were generated.")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_card(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a slide-ready final takeaway card."""
    sm_row = next(row for row in rows if row["model"] == "SM-FNO2D v3 optimized")
    persistence_row = next(row for row in rows if row["model"] == "Persistence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.2, 8.0), facecolor=FIG_FACE)
    ax.axis("off")
    ax.text(
        0.04,
        0.88,
        "Diagnostic takeaway",
        color=TEXT_COLOR,
        fontsize=28,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.75,
        "Optimized SM-FNO2D v3 learned dynamics\nbeyond last-frame persistence",
        color="#34d399",
        fontsize=20,
        weight="bold",
        linespacing=1.2,
        transform=ax.transAxes,
    )
    bullets = [
        "Dynamic Navier-Stokes2D fixture, full held-out 36-step rollouts",
        "FNO residual base + gated SSM correction",
        "Delta-aware and persistence-aware rollout training",
        "Diagnostic result only; no broad benchmark claim",
    ]
    for index, bullet in enumerate(bullets):
        ax.text(
            0.07,
            0.50 - index * 0.085,
            f"- {bullet}",
            color=MUTED_TEXT,
            fontsize=14,
            transform=ax.transAxes,
        )
    metric_lines = [
        ("SM-FNO2D v3 skill mean", f"{sm_row['persistence_skill_mean']:.3f}"),
        ("SM-FNO2D v3 step-36 skill", f"{sm_row['persistence_skill_step36']:.3f}"),
        ("Persistence mean Rel L2", f"{persistence_row['rollout_rel_l2_mean']:.3f}"),
        ("SM-FNO2D v3 mean Rel L2", f"{sm_row['rollout_rel_l2_mean']:.3f}"),
    ]
    for index, (label, value) in enumerate(metric_lines):
        x = 0.62 + (index % 2) * 0.18
        y = 0.48 - (index // 2) * 0.23
        box = plt.Rectangle((x, y), 0.16, 0.16, facecolor="#111827", edgecolor="#334155")
        ax.add_patch(box)
        ax.text(x + 0.02, y + 0.095, value, color=TEXT_COLOR, fontsize=22, weight="bold")
        ax.text(x + 0.02, y + 0.045, label, color=MUTED_TEXT, fontsize=8.5)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=FIG_FACE)
    plt.close(fig)


def main() -> None:
    """Generate all M18 presentation artifacts."""
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    three_d_dir = args.figures_dir / "3d_style"
    three_d_dir.mkdir(parents=True, exist_ok=True)

    fno_artifact = _load_rollout_artifact(
        key="fno",
        label="FNO2D",
        path=args.fno_artifact,
        metrics_path=args.fno_metrics,
    )
    sm_artifact = _load_rollout_artifact(
        key="sm_fno",
        label="SM-FNO2D v3 optimized",
        path=args.sm_fno_artifact,
        metrics_path=args.sm_fno_metrics,
    )
    transformer_artifact = _load_optional_rollout_artifact(
        key="transformer",
        label="Transformer2D",
        path=args.transformer_artifact,
        metrics_path=args.transformer_metrics,
    )
    if fno_artifact.targets.shape != sm_artifact.targets.shape:
        raise ValueError("FNO2D and SM-FNO2D artifacts must share target shape.")
    if (
        transformer_artifact is not None
        and transformer_artifact.targets.shape != sm_artifact.targets.shape
    ):
        raise ValueError("Transformer2D and SM-FNO2D artifacts must share target shape.")

    targets = sm_artifact.targets
    persistence = sm_artifact.persistence
    sample_index = _select_sample_index(targets, persistence, args.sample_index)
    fno_prediction = fno_artifact.predictions
    transformer_prediction = (
        transformer_artifact.predictions if transformer_artifact is not None else None
    )
    sm_prediction = sm_artifact.predictions
    persistence_prediction = persistence

    rows = [
        _summarize_model(
            sm_artifact,
            persistence_prediction,
            label="Persistence",
        ),
        _summarize_model(fno_artifact, fno_prediction, label="FNO2D"),
    ]
    if transformer_artifact is not None and transformer_prediction is not None:
        rows.append(
            _summarize_model(
                transformer_artifact,
                transformer_prediction,
                label="Transformer2D",
            )
        )
    rows.append(_summarize_model(sm_artifact, sm_prediction, label="SM-FNO2D v3 optimized"))
    _write_comparison_tables(rows, args.tables_dir)

    persistence_error = _relative_l2_series(persistence_prediction, targets)
    true_change = true_step_change_relative_l2(sm_artifact.inputs, targets)
    fno_error = _relative_l2_series(fno_prediction, targets)
    transformer_error = (
        _relative_l2_series(transformer_prediction, targets)
        if transformer_prediction is not None
        else None
    )
    sm_error = _relative_l2_series(sm_prediction, targets)
    fno_delta = delta_prediction_relative_l2(
        fno_prediction,
        targets,
        sm_artifact.inputs[:, -1],
    )
    transformer_delta = (
        delta_prediction_relative_l2(
            transformer_prediction,
            targets,
            sm_artifact.inputs[:, -1],
        )
        if transformer_prediction is not None
        else None
    )
    sm_delta = delta_prediction_relative_l2(
        sm_prediction,
        targets,
        sm_artifact.inputs[:, -1],
    )
    fno_skill = persistence_skill_score(fno_error, persistence_error)
    transformer_skill = (
        persistence_skill_score(transformer_error, persistence_error)
        if transformer_error is not None
        else None
    )
    sm_skill = persistence_skill_score(sm_error, persistence_error)
    rollout_series = {"Persistence": persistence_error.tolist(), "FNO2D": fno_error.tolist()}
    skill_series = {"FNO2D": fno_skill.tolist()}
    delta_series = {"Persistence": np.ones_like(sm_delta).tolist(), "FNO2D": fno_delta.tolist()}
    if (
        transformer_error is not None
        and transformer_delta is not None
        and transformer_skill is not None
    ):
        rollout_series["Transformer2D"] = transformer_error.tolist()
        skill_series["Transformer2D"] = transformer_skill.tolist()
        delta_series["Transformer2D"] = transformer_delta.tolist()
    rollout_series["SM-FNO2D v3 optimized"] = sm_error.tolist()
    skill_series["SM-FNO2D v3 optimized"] = sm_skill.tolist()
    delta_series["SM-FNO2D v3 optimized"] = sm_delta.tolist()

    _save_line_plot(
        args.figures_dir / "m18_rollout_relative_l2_comparison.png",
        title="Dynamic fixture: rollout error vs persistence",
        ylabel="Relative L2 error",
        series=rollout_series,
    )
    _save_line_plot(
        args.figures_dir / "m18_persistence_normalized_skill.png",
        title="Dynamic fixture: persistence-normalized skill",
        ylabel="Skill = 1 - model error / persistence error",
        series=skill_series,
        horizontal_zero=True,
    )
    _save_line_plot(
        args.figures_dir / "m18_heldout_temporal_variation.png",
        title="Dynamic fixture: held-out temporal variation",
        ylabel="Relative L2",
        series={"Persistence": persistence_error.tolist()},
        extra_series={"True step change": true_change.tolist()},
    )
    _save_line_plot(
        args.figures_dir / "m18_delta_prediction_error.png",
        title="Dynamic fixture: delta-prediction error",
        ylabel="Delta relative L2",
        series=delta_series,
    )
    _save_cost_performance(rows, args.figures_dir / "m18_cost_vs_diagnostic_error.png")
    _write_summary_card(rows, args.figures_dir / "m18_presentation_summary_card.png")

    target_sequence = _to_field(targets, sample_index)
    persistence_sequence = _to_field(persistence_prediction, sample_index)
    fno_sequence = _to_field(fno_prediction, sample_index)
    transformer_sequence = (
        _to_field(transformer_prediction, sample_index)
        if transformer_prediction is not None
        else None
    )
    sm_sequence = _to_field(sm_prediction, sample_index)
    all_sequence_values = [target_sequence, persistence_sequence, fno_sequence, sm_sequence]
    if transformer_sequence is not None:
        all_sequence_values.append(transformer_sequence)
    all_sequence = np.stack(all_sequence_values, axis=0)
    field_abs = float(max(np.abs(all_sequence).max(), 1e-6))
    error_max = float(
        max(
            np.abs(persistence_sequence - target_sequence).max(),
            np.abs(fno_sequence - target_sequence).max(),
            (
                np.abs(transformer_sequence - target_sequence).max()
                if transformer_sequence is not None
                else 0.0
            ),
            np.abs(sm_sequence - target_sequence).max(),
            1e-6,
        )
    )
    sequence_by_label = {
        "persistence": ("Persistence", persistence_sequence),
        "fno": ("FNO2D", fno_sequence),
        "v3": ("SM-FNO2D v3 optimized", sm_sequence),
    }
    if transformer_sequence is not None:
        sequence_by_label["transformer"] = ("Transformer2D", transformer_sequence)
    for step in args.steps:
        step_index = min(max(step, 1), target_sequence.shape[0]) - 1
        for key, (label, sequence) in sequence_by_label.items():
            _save_triptych_panel(
                target=target_sequence[step_index],
                prediction=sequence[step_index],
                output_path=args.figures_dir / f"m18_{key}_step{step:02d}_panels.png",
                title=f"{label}: rollout step {step}",
                field_abs=field_abs,
                error_max=error_max,
            )
        if step == 36:
            _save_step_comparison_panel(
                target=target_sequence[step_index],
                predictions={
                    "Persistence": persistence_sequence[step_index],
                    "FNO2D": fno_sequence[step_index],
                    **(
                        {"Transformer2D": transformer_sequence[step_index]}
                        if transformer_sequence is not None
                        else {}
                    ),
                    "SM-FNO2D v3 optimized": sm_sequence[step_index],
                },
                output_path=args.figures_dir / "m18_step36_model_comparison.png",
                step=step,
            )
        plot_vorticity_surface_triptych(
            target=target_sequence[step_index],
            prediction=sm_sequence[step_index],
            output_path=args.figures_dir / f"m18_v3_surface_step{step:02d}.png",
            title=f"SM-FNO2D v3 optimized 3D-style diagnostic - step {step}",
            dpi=180,
        )

    _save_triptych_gif(
        targets=target_sequence,
        predictions=sm_sequence,
        output_path=args.figures_dir / "m18_v3_rollout.gif",
        label="SM-FNO2D v3 optimized",
        fps=args.fps,
        max_frames=args.max_gif_frames,
    )
    _save_triptych_gif(
        targets=target_sequence,
        predictions=fno_sequence,
        output_path=args.figures_dir / "m18_fno_rollout.gif",
        label="FNO2D",
        fps=args.fps,
        max_frames=args.max_gif_frames,
    )
    _save_triptych_gif(
        targets=target_sequence,
        predictions=persistence_sequence,
        output_path=args.figures_dir / "m18_persistence_rollout.gif",
        label="Persistence",
        fps=args.fps,
        max_frames=args.max_gif_frames,
    )
    comparison_gif_name = (
        "m18_true_persistence_fno_transformer_v3_comparison.gif"
        if transformer_sequence is not None
        else "m18_true_persistence_fno_v3_comparison.gif"
    )
    _save_model_comparison_gif(
        sequences={
            "True": target_sequence,
            "Persistence": persistence_sequence,
            "FNO2D": fno_sequence,
            **(
                {"Transformer2D": transformer_sequence}
                if transformer_sequence is not None
                else {}
            ),
            "SM-FNO2D v3": sm_sequence,
        },
        output_path=args.figures_dir / comparison_gif_name,
        fps=args.fps,
        max_frames=args.max_gif_frames,
    )
    if transformer_sequence is not None:
        _save_triptych_gif(
            targets=target_sequence,
            predictions=transformer_sequence,
            output_path=args.figures_dir / "m18_transformer_rollout.gif",
            label="Transformer2D",
            fps=args.fps,
            max_frames=args.max_gif_frames,
        )

    surface_sequences = {
        "true": ("True target", target_sequence),
        "v3": ("SM-FNO2D v3 optimized", sm_sequence),
        "fno": ("FNO2D", fno_sequence),
        "persistence": ("Persistence", persistence_sequence),
    }
    if transformer_sequence is not None:
        surface_sequences["transformer"] = ("Transformer2D", transformer_sequence)
    for key, (label, sequence) in surface_sequences.items():
        save_vorticity_surface_rollout_animation(
            sequence,
            three_d_dir / f"m18_3d_{key}_rollout.gif",
            max_frames=args.surface_max_frames,
            fps=args.surface_fps,
            dpi=args.surface_dpi,
            title=f"{label} 3D-style 2D vorticity",
            panel_title=label,
            field_abs=field_abs,
        )
    comparison_sequences = {
        "True target": target_sequence,
        "Persistence": persistence_sequence,
        "FNO2D": fno_sequence,
        **({"Transformer2D": transformer_sequence} if transformer_sequence is not None else {}),
        "SM-FNO2D v3": sm_sequence,
    }
    surface_comparison_name = (
        "m18_3d_true_persistence_fno_transformer_v3_comparison.gif"
        if transformer_sequence is not None
        else "m18_3d_true_persistence_fno_v3_comparison.gif"
    )
    save_vorticity_surface_comparison_animation(
        comparison_sequences,
        three_d_dir / surface_comparison_name,
        max_frames=args.surface_max_frames,
        fps=args.surface_fps,
        dpi=args.surface_dpi,
        title="3D-style rendering of 2D vorticity rollout comparison",
        field_abs=field_abs,
    )

    mp4_status = []
    if args.write_mp4:
        if animation.writers.is_available("ffmpeg"):
            save_vorticity_surface_rollout_animation(
                sm_sequence,
                three_d_dir / "m18_3d_v3_rollout.mp4",
                max_frames=args.surface_max_frames,
                fps=args.surface_fps,
                dpi=args.surface_dpi,
                title="SM-FNO2D v3 3D-style 2D vorticity",
                panel_title="SM-FNO2D v3 optimized",
                field_abs=field_abs,
            )
            save_vorticity_surface_comparison_animation(
                comparison_sequences,
                three_d_dir / "m18_3d_true_persistence_fno_v3_comparison.mp4",
                max_frames=args.surface_max_frames,
                fps=args.surface_fps,
                dpi=args.surface_dpi,
                title="3D-style rendering of 2D vorticity rollout comparison",
                field_abs=field_abs,
            )
            mp4_status.extend(
                [
                    f"MP4 generated: `{three_d_dir / 'm18_3d_v3_rollout.mp4'}`",
                    "MP4 generated: "
                    f"`{three_d_dir / 'm18_3d_true_persistence_fno_v3_comparison.mp4'}`",
                ]
            )
        else:
            mp4_status.append("MP4 skipped: matplotlib ffmpeg writer is unavailable.")
    else:
        mp4_status.append("MP4 export was not requested; GIF artifacts were generated.")

    manifest = {
        "notice": NOTICE,
        "selected_sample_index": sample_index,
        "rows": rows,
        "figures_dir": str(args.figures_dir),
        "three_d_dir": str(three_d_dir),
        "tables_dir": str(args.tables_dir),
        "transformer_included": transformer_artifact is not None,
        "mp4_status": mp4_status,
    }
    manifest_path = args.tables_dir / "m18_artifact_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    _write_artifact_index(
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        three_d_dir=three_d_dir,
        selected_sample=sample_index,
        include_transformer=transformer_artifact is not None,
        mp4_status=mp4_status,
    )

    print(f"[m18] wrote figures to {args.figures_dir}")
    print(f"[m18] wrote 3D-style animations to {three_d_dir}")
    print(f"[m18] wrote tables to {args.tables_dir}")
    print(f"[m18] selected held-out sample {sample_index}")
    for row in rows:
        print(
            "[m18] "
            f"{row['model']}: mean_rel_l2={row['rollout_rel_l2_mean']:.6f}, "
            f"skill_mean={row['persistence_skill_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
