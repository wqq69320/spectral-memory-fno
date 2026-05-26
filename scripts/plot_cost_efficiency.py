"""Plot protocol-scale cost-efficiency summaries from aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sm_fno.visualization import plots as plot_helpers


plt = plot_helpers.plt


NOTICE = (
    "Cost-efficiency plots are protocol-scale analysis artifacts only; "
    "they are not final benchmark claims or model rankings."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot cost-efficiency aggregate metrics.")
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("results") / "tables" / "m9_cost_efficiency_aggregate.json",
        help="Aggregate JSON produced by scripts/aggregate_cost_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "figures" / "m9_cost_efficiency",
        help="Directory for generated PNG plots.",
    )
    return parser.parse_args()


def _metric_mean(group: dict[str, Any], metric_key: str) -> float | None:
    """Return a group's metric mean when present."""
    summary = group.get("metrics", {}).get(metric_key, {})
    value = summary.get("mean")
    return None if value is None else float(value)


def _series_label(group: dict[str, Any]) -> str:
    """Build a stable display label for one dataset/model series."""
    return f"{group['dataset_name']} / {group['model_name']}"


def plot_metric_by_horizon(
    aggregate: dict[str, Any],
    *,
    metric_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot one aggregate metric against rollout horizon."""
    series: dict[str, list[tuple[int, float]]] = {}
    for group in aggregate.get("groups", []):
        value = _metric_mean(group, metric_key)
        if value is None:
            continue
        series.setdefault(_series_label(group), []).append((int(group["rollout_steps"]), value))

    if not series:
        raise ValueError(f"No aggregate groups contained metric {metric_key}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, points in sorted(series.items()):
        points = sorted(points)
        ax.plot(
            [horizon for horizon, _ in points],
            [value for _, value in points],
            marker="o",
            label=label,
        )
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_error_vs_time(aggregate: dict[str, Any], output_path: Path) -> None:
    """Plot rollout error against normalized rollout time."""
    points: list[tuple[float, float, str, int]] = []
    for group in aggregate.get("groups", []):
        error = _metric_mean(group, "rollout_relative_l2")
        seconds_per_step = _metric_mean(group, "rollout_seconds_per_step")
        if error is None or seconds_per_step is None:
            continue
        points.append((seconds_per_step, error, _series_label(group), int(group["rollout_steps"])))

    if not points:
        raise ValueError("No aggregate groups contained rollout error and timing metrics.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seconds_per_step, error, label, horizon in points:
        ax.scatter(seconds_per_step, error, s=35)
        ax.annotate(f"{label} h{horizon}", (seconds_per_step, error), fontsize=6)
    ax.set_xlabel("Rollout seconds per step")
    ax.set_ylabel("Rollout relative L2")
    ax.set_title("Rollout error vs normalized rollout time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_plots(aggregate: dict[str, Any], output_dir: Path) -> list[Path]:
    """Write all M9 cost-efficiency plots and return their paths."""
    plot_paths = [
        output_dir / "rollout_relative_l2_by_horizon.png",
        output_dir / "rollout_seconds_per_step_by_horizon.png",
        output_dir / "rollout_error_vs_seconds_per_step.png",
    ]
    plot_metric_by_horizon(
        aggregate,
        metric_key="rollout_relative_l2",
        ylabel="Rollout relative L2",
        output_path=plot_paths[0],
    )
    plot_metric_by_horizon(
        aggregate,
        metric_key="rollout_seconds_per_step",
        ylabel="Rollout seconds per step",
        output_path=plot_paths[1],
    )
    plot_error_vs_time(aggregate, plot_paths[2])
    return plot_paths


def main() -> None:
    """Write plots from an aggregate cost metrics JSON."""
    args = parse_args()
    with args.aggregate.open("r", encoding="utf-8") as file:
        aggregate = json.load(file)
    for path in write_plots(aggregate, args.output_dir):
        print(f"[plot_cost_efficiency] wrote {path}")
    print(f"[plot_cost_efficiency] notice: {NOTICE}")


if __name__ == "__main__":
    main()
