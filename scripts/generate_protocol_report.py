"""Generate a protocol summary report from aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NOTICE = (
    "This report summarizes smoke and repeated-seed protocol outputs only. "
    "It does not rank models or make benchmark claims."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate an M6 protocol report.")
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("results") / "tables" / "aggregate_metrics.json",
        help="Aggregate metrics JSON produced by scripts/aggregate_metrics.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "tables" / "protocol_summary.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def _format_metric(summary: dict[str, Any]) -> str:
    """Format a metric summary for display."""
    if summary.get("mean") is None:
        return "NA"
    return f"{summary['mean']:.6f} +/- {summary['std']:.6f}"


def render_protocol_report(aggregate: dict[str, Any]) -> str:
    """Render aggregate metrics as a protocol-focused Markdown report."""
    lines = [
        "# Protocol Summary",
        "",
        NOTICE,
        "",
        "## Aggregate Sanity Metrics",
        "",
        "| Experiment | Run Type | Count | Seeds | MSE | Relative L2 | Rollout Relative L2 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for group in aggregate.get("groups", []):
        metrics = group["metrics"]
        seeds = ", ".join(str(seed) for seed in group.get("seeds", [])) or "NA"
        lines.append(
            "| "
            f"`{group['base_experiment']}` | "
            f"`{group['run_type']}` | "
            f"{group['count']} | "
            f"{seeds} | "
            f"{_format_metric(metrics['mse'])} | "
            f"{_format_metric(metrics['relative_l2'])} | "
            f"{_format_metric(metrics['rollout_relative_l2'])} |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Smoke and repeated-seed runs are CPU-friendly protocol checks.",
            "- Single small Heat1D datasets do not support performance claims.",
            "- Timing values are wall-clock measurements from small CPU runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the protocol report."""
    args = parse_args()
    with args.aggregate.open("r", encoding="utf-8") as file:
        aggregate = json.load(file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_protocol_report(aggregate), encoding="utf-8")
    print(f"[generate_protocol_report] wrote {args.output}")


if __name__ == "__main__":
    main()
