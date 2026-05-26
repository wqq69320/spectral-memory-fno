"""Generate a protocol-scale cost-efficiency Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NOTICE = (
    "This report summarizes protocol-scale CPU analysis outputs only. "
    "It does not rank models or make final benchmark claims."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a cost-efficiency report.")
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("results") / "tables" / "m9_cost_efficiency_aggregate.json",
        help="Aggregate metrics JSON produced by scripts/aggregate_cost_metrics.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "tables" / "m9_cost_efficiency_report.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def _format_metric(summary: dict[str, Any]) -> str:
    """Format a metric summary for display."""
    if summary.get("mean") is None:
        return "NA"
    return f"{summary['mean']:.6f} +/- {summary['std']:.6f}"


def render_cost_efficiency_report(aggregate: dict[str, Any]) -> str:
    """Render aggregate cost metrics as a protocol-focused Markdown report."""
    lines = [
        "# Cost-Efficiency Protocol Report",
        "",
        NOTICE,
        "",
        "## Aggregate Protocol Metrics",
        "",
        "| Dataset | Model | Horizon | Seeds | Parameters | Rollout Relative L2 | "
        "Rollout Seconds/Step |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for group in aggregate.get("groups", []):
        metrics = group["metrics"]
        seeds = ", ".join(str(seed) for seed in group.get("seeds", [])) or "NA"
        lines.append(
            "| "
            f"`{group['dataset_name']}` | "
            f"`{group['model_name']}` | "
            f"{group['rollout_steps']} | "
            f"{seeds} | "
            f"{_format_metric(metrics['model_parameter_count'])} | "
            f"{_format_metric(metrics['rollout_relative_l2'])} | "
            f"{_format_metric(metrics['rollout_seconds_per_step'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- These are small CPU protocol runs for analysis tooling validation.",
            "- Timing values are local wall-clock measurements and can vary by machine load.",
            "- The outputs are appropriate for checking reporting mechanics, not for ranking models.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the M9 generated report."""
    args = parse_args()
    with args.aggregate.open("r", encoding="utf-8") as file:
        aggregate = json.load(file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_cost_efficiency_report(aggregate), encoding="utf-8")
    print(f"[generate_cost_efficiency_report] wrote {args.output}")


if __name__ == "__main__":
    main()
