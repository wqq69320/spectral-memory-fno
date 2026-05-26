"""Aggregate protocol-scale cost and rollout metrics."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


COST_METRIC_KEYS = (
    "mse",
    "relative_l2",
    "rollout_relative_l2",
    "rollout_inference_seconds",
    "rollout_seconds_per_step",
    "rollout_seconds_per_example",
    "rollout_seconds_per_example_per_step",
    "one_step_inference_seconds",
    "one_step_seconds_per_forward",
    "one_step_seconds_per_example",
    "model_parameter_count",
    "model_trainable_parameter_count",
)
NOTICE = (
    "Cost-efficiency outputs are protocol-scale analysis artifacts only; "
    "they are not final benchmark claims or model rankings."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate M9 cost-efficiency metrics.")
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob for horizon-sweep evaluation metrics JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "tables" / "m9_cost_efficiency_aggregate.json",
        help="Path to write aggregate JSON.",
    )
    return parser.parse_args()


def _float_or_none(value: Any) -> float | None:
    """Convert values to floats while preserving missing values."""
    if value is None:
        return None
    return float(value)


def _infer_dataset_name(metrics: dict[str, Any]) -> str:
    """Infer dataset name from metadata when older metrics omit it."""
    if metrics.get("dataset_name"):
        return str(metrics["dataset_name"])
    data_path = str(metrics.get("data_path", "unknown"))
    if "burgers1d" in data_path:
        return "burgers1d"
    if "heat1d" in data_path:
        return "heat1d"
    if "navier_stokes2d" in data_path:
        return "navier_stokes2d"
    return "unknown"


def load_cost_metric_records(input_glob: str) -> list[dict[str, Any]]:
    """Load cost metric records matching the provided glob."""
    paths = sorted(glob.glob(input_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No metric files matched: {input_glob}")

    records: list[dict[str, Any]] = []
    for path_text in paths:
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        records.append(
            {
                "path": str(path),
                "experiment": str(metrics.get("experiment", path.stem.replace("_eval_metrics", ""))),
                "base_experiment": str(metrics.get("base_experiment", "")),
                "analysis_name": str(metrics.get("analysis_name", "")),
                "run_type": str(metrics.get("run_type", "")),
                "seed": metrics.get("seed"),
                "dataset_name": _infer_dataset_name(metrics),
                "model_name": str(metrics.get("model_name", "unknown")),
                "rollout_steps": int(metrics.get("rollout_steps", 0)),
                "metrics": {
                    metric_key: _float_or_none(metrics.get(metric_key))
                    for metric_key in COST_METRIC_KEYS
                },
            }
        )
    return records


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    """Compute mean and sample standard deviation for one metric."""
    if not values:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_cost_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate cost records by dataset, model, base experiment, and horizon."""
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["dataset_name"]),
            str(record["model_name"]),
            str(record["base_experiment"]),
            str(record["run_type"]),
            int(record["rollout_steps"]),
        )
        grouped.setdefault(key, []).append(record)

    groups: list[dict[str, Any]] = []
    for (dataset_name, model_name, base_experiment, run_type, rollout_steps), group_records in sorted(
        grouped.items()
    ):
        metric_summary: dict[str, dict[str, float | int | None]] = {}
        for metric_key in COST_METRIC_KEYS:
            values = [
                float(record["metrics"][metric_key])
                for record in group_records
                if record["metrics"][metric_key] is not None
            ]
            metric_summary[metric_key] = summarize_values(values)

        groups.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "base_experiment": base_experiment,
                "run_type": run_type,
                "rollout_steps": rollout_steps,
                "count": len(group_records),
                "seeds": sorted(
                    int(record["seed"])
                    for record in group_records
                    if record.get("seed") is not None
                ),
                "metric_files": [record["path"] for record in group_records],
                "metrics": metric_summary,
            }
        )

    return {"notice": NOTICE, "groups": groups, "records": records}


def write_cost_aggregate_json(aggregate: dict[str, Any], output_path: Path) -> None:
    """Write aggregate cost data as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=2)


def _format_summary(summary: dict[str, float | int | None]) -> str:
    """Format a mean/std pair for Markdown."""
    if summary["mean"] is None:
        return "NA"
    return f"{summary['mean']:.10f} +/- {summary['std']:.10f}"


def write_cost_aggregate_markdown(aggregate: dict[str, Any], output_path: Path) -> None:
    """Write aggregate cost data as a Markdown table."""
    markdown_path = output_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cost-Efficiency Aggregate Metrics",
        "",
        NOTICE,
        "",
        "| Dataset | Model | Horizon | Count | Seeds | Parameters | Rollout Relative L2 | "
        "Rollout Seconds/Step | Rollout Seconds/Example/Step |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for group in aggregate["groups"]:
        metrics = group["metrics"]
        seeds = ", ".join(str(seed) for seed in group["seeds"]) or "NA"
        lines.append(
            "| "
            f"`{group['dataset_name']}` | "
            f"`{group['model_name']}` | "
            f"{group['rollout_steps']} | "
            f"{group['count']} | "
            f"{seeds} | "
            f"{_format_summary(metrics['model_parameter_count'])} | "
            f"{_format_summary(metrics['rollout_relative_l2'])} | "
            f"{_format_summary(metrics['rollout_seconds_per_step'])} | "
            f"{_format_summary(metrics['rollout_seconds_per_example_per_step'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Aggregate metric JSON files and write JSON plus Markdown outputs."""
    args = parse_args()
    records = load_cost_metric_records(args.input_glob)
    aggregate = aggregate_cost_records(records)
    write_cost_aggregate_json(aggregate, args.output)
    write_cost_aggregate_markdown(aggregate, args.output)
    print(f"[aggregate_cost_metrics] wrote {args.output}")
    print(f"[aggregate_cost_metrics] wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
