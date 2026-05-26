"""Aggregate experiment metric JSON files."""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "mse",
    "relative_l2",
    "rollout_relative_l2",
    "one_step_inference_seconds",
    "rollout_inference_seconds",
)
SEED_SUFFIX_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")
NOTICE = (
    "Smoke and repeated-seed metrics are protocol sanity outputs only; "
    "they are not benchmark claims or model rankings."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate SM-FNO metric JSON files.")
    parser.add_argument("--input-glob", required=True, help="Glob for evaluation metrics JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "tables" / "aggregate_metrics.json",
        help="Path to write aggregate JSON.",
    )
    return parser.parse_args()


def _float_or_none(value: Any) -> float | None:
    """Convert metric values to floats while preserving missing values."""
    if value is None:
        return None
    return float(value)


def load_metric_records(input_glob: str) -> list[dict[str, Any]]:
    """Load metric records matching the provided glob."""
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No metric files matched: {input_glob}")

    records: list[dict[str, Any]] = []
    for path_text in paths:
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        experiment = str(metrics.get("experiment", path.stem.replace("_eval_metrics", "")))
        base_experiment = str(metrics.get("base_experiment", ""))
        run_type = str(metrics.get("run_type", ""))
        seed = metrics.get("seed")

        match = SEED_SUFFIX_RE.match(experiment)
        if not base_experiment:
            base_experiment = match.group("base") if match else experiment
        if not run_type:
            run_type = "repeated_seed" if match else "single"
        if seed is None and match:
            seed = int(match.group("seed"))

        records.append(
            {
                "path": str(path),
                "experiment": experiment,
                "base_experiment": base_experiment,
                "run_type": run_type,
                "seed": seed,
                "metrics": {
                    metric_key: _float_or_none(metrics.get(metric_key))
                    for metric_key in METRIC_KEYS
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


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metric records by base experiment and run type."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["base_experiment"]), str(record["run_type"]))
        grouped.setdefault(key, []).append(record)

    groups: list[dict[str, Any]] = []
    for (base_experiment, run_type), group_records in sorted(grouped.items()):
        metric_summary: dict[str, dict[str, float | int | None]] = {}
        for metric_key in METRIC_KEYS:
            values = [
                float(record["metrics"][metric_key])
                for record in group_records
                if record["metrics"][metric_key] is not None
            ]
            metric_summary[metric_key] = summarize_values(values)

        groups.append(
            {
                "base_experiment": base_experiment,
                "run_type": run_type,
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


def write_aggregate_json(aggregate: dict[str, Any], output_path: Path) -> None:
    """Write aggregate data as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=2)


def _format_summary(summary: dict[str, float | int | None]) -> str:
    """Format a mean/std pair for Markdown."""
    if summary["mean"] is None:
        return "NA"
    return f"{summary['mean']:.10f} +/- {summary['std']:.10f}"


def write_aggregate_markdown(aggregate: dict[str, Any], output_path: Path) -> None:
    """Write aggregate data as a Markdown table."""
    markdown_path = output_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Aggregate Metrics",
        "",
        NOTICE,
        "",
        "| Experiment | Run Type | Count | Seeds | MSE | Relative L2 | "
        "Rollout Relative L2 | One-step Seconds | Rollout Seconds |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in aggregate["groups"]:
        metrics = group["metrics"]
        seeds = ", ".join(str(seed) for seed in group["seeds"]) or "NA"
        lines.append(
            "| "
            f"`{group['base_experiment']}` | "
            f"`{group['run_type']}` | "
            f"{group['count']} | "
            f"{seeds} | "
            f"{_format_summary(metrics['mse'])} | "
            f"{_format_summary(metrics['relative_l2'])} | "
            f"{_format_summary(metrics['rollout_relative_l2'])} | "
            f"{_format_summary(metrics['one_step_inference_seconds'])} | "
            f"{_format_summary(metrics['rollout_inference_seconds'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Aggregate metric JSON files and write JSON plus Markdown outputs."""
    args = parse_args()
    records = load_metric_records(args.input_glob)
    aggregate = aggregate_records(records)
    write_aggregate_json(aggregate, args.output)
    write_aggregate_markdown(aggregate, args.output)
    print(f"[aggregate_metrics] wrote {args.output}")
    print(f"[aggregate_metrics] wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
