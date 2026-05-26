# M9 Cost-Efficiency Analysis Plan

## Objective

Add a config-driven analysis layer that measures long-horizon rollout error and
lightweight compute-cost metrics for Heat1D and Burgers1D across FNO1D, SM-FNO,
and Transformer1D. M9 outputs are protocol-scale analysis artifacts only and do
not support benchmark claims or model rankings.

## Scope

- Extend normal training and evaluation metrics with parameter counts and
  normalized inference timing.
- Run a CPU-friendly horizon sweep over shared Heat1D and Burgers1D smoke
  configs for FNO1D, SM-FNO, and Transformer1D.
- Evaluate rollout horizons `5`, `10`, and `20` using repeated seeds `0` and
  `1`, training once per base config and seed.
- Aggregate cost metrics by dataset, model, base experiment, run type, horizon,
  and seed.
- Generate protocol-scale Markdown tables and cost-efficiency plots.
- Keep generated configs, metrics, checkpoints, predictions, figures, and local
  reports under ignored output paths.

## Done Criteria

- [x] `scripts/train.py` logs model parameter counts.
- [x] `scripts/evaluate.py` logs model parameter counts, one-step timing, rollout
  timing, rollout seconds per step, and rollout seconds per example per step.
- [x] `configs/analysis/m9_cost_efficiency_smoke.yaml` defines the CPU-friendly
  Heat1D/Burgers1D model/horizon/seed sweep.
- [x] `scripts/run_horizon_sweep.py` expands the analysis config into clear
  per-seed/per-horizon configs and runs train/evaluate.
- [x] `scripts/aggregate_cost_metrics.py` aggregates M9 cost and rollout metrics
  to JSON and Markdown.
- [x] `scripts/plot_cost_efficiency.py` writes cost-efficiency plots from the
  aggregate JSON.
- [x] `scripts/generate_cost_efficiency_report.py` writes a generated
  protocol-scale report from aggregate metrics.
- [x] Tests cover parameter-count/timing helpers, horizon-sweep expansion,
  cost aggregation, and plot/report helpers.
- [x] README current status mentions M9 without making performance claims.
- [x] Verification commands pass on current code.
- [x] Generated M9 artifacts are confirmed git-ignored.
- [x] `docs/reports/M9_COST_EFFICIENCY_REPORT.md` records command status,
  artifact paths, limitations, and no benchmark claims.

## Required Verification

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/heat1d.yaml
PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/burgers1d.yaml
PYTHONPATH=src python3 scripts/run_horizon_sweep.py --config configs/analysis/m9_cost_efficiency_smoke.yaml
PYTHONPATH=src python3 scripts/aggregate_cost_metrics.py --input-glob "results/tables/m9_cost_efficiency/*_eval_metrics.json" --output results/tables/m9_cost_efficiency_aggregate.json
PYTHONPATH=src python3 scripts/plot_cost_efficiency.py --aggregate results/tables/m9_cost_efficiency_aggregate.json --output-dir results/figures/m9_cost_efficiency
PYTHONPATH=src python3 scripts/generate_cost_efficiency_report.py --aggregate results/tables/m9_cost_efficiency_aggregate.json --output results/tables/m9_cost_efficiency_report.md
git diff --check
```

## Non-Goals

- Do not tune models for performance.
- Do not add larger datasets or final benchmark settings.
- Do not claim that one model is better, faster, or more cost-efficient than
  another based on these protocol-scale runs.
