# M6 Fair Comparison Protocol

## Objective

Harden the Heat1D comparison pipeline so `MLPBaseline`, `FNO1D`,
`SpectralMemoryFNO1D`, and `Transformer1DBaseline` comparisons are fair,
repeatable, and reportable.

M6 is protocol-hardening work only. Smoke and repeated-seed outputs are sanity
checks for the experiment pipeline, not benchmark results or model rankings.

## Shared Heat1D Protocol

All models in a fair Heat1D comparison must use:

- The same generated dataset path and data-generation config.
- The same trajectory-level train/validation/test split ratios.
- The same split seed for a given repeated-seed run.
- The same `input_steps`.
- The same `pred_steps`.
- The same `rollout_steps`.
- The same training budget where appropriate: epochs, batch size, optimizer,
  learning rate, weight decay, gradient clipping, and device.
- The same evaluation script and metric definitions.
- Repeated seeds before interpreting variability.

Smoke configs are intentionally small and CPU-friendly. They are used to verify
that the protocol runs end to end. They must not be cited as evidence that one
model is better than another.

## Implemented Scope

- Repeated-seed execution:
  `scripts/run_repeated_seeds.py`
- Metric aggregation:
  `scripts/aggregate_metrics.py`
- Protocol summary report generation:
  `scripts/generate_protocol_report.py`
- Shared smoke protocol configs:
  - `configs/experiment/heat1d_mlp_smoke.yaml`
  - `configs/experiment/heat1d_fno_smoke.yaml`
  - `configs/experiment/heat1d_sm_fno_smoke.yaml`
  - `configs/experiment/heat1d_transformer_smoke.yaml`
- Longer rollout protocol-validation configs:
  - `configs/experiment/heat1d_mlp_rollout20.yaml`
  - `configs/experiment/heat1d_fno_rollout20.yaml`
  - `configs/experiment/heat1d_sm_fno_rollout20.yaml`
  - `configs/experiment/heat1d_transformer_rollout20.yaml`

The rollout-20 configs are protocol-validation configs. They are not final
benchmark configs.

## Repeated-Seed Execution

The repeated-seed runner accepts one base experiment config and one or more
seeds:

```bash
PYTHONPATH=src python3 scripts/run_repeated_seeds.py \
  --config configs/experiment/heat1d_fno_smoke.yaml \
  --seeds 0 1
```

For each seed, the runner expands the base config with:

- A seed-specific experiment name.
- `base_experiment` metadata.
- `run_type: repeated_seed`.
- Seed-specific checkpoint, train-metric, eval-metric, and prediction paths.

Per-seed metrics are written with names such as:

- `results/tables/heat1d_fno_smoke_seed0_eval_metrics.json`
- `results/tables/heat1d_fno_smoke_seed1_eval_metrics.json`

## Metric Aggregation

The aggregation script reads evaluation metric JSON files and writes:

- `results/tables/aggregate_metrics.json`
- `results/tables/aggregate_metrics.md`

It computes mean and sample standard deviation for:

- `mse`
- `relative_l2`
- `rollout_relative_l2`
- `one_step_inference_seconds`
- `rollout_inference_seconds`

Single-run groups use `0.0` standard deviation.

## Protocol Report

The protocol report generator reads `aggregate_metrics.json` and writes a
Markdown summary:

- `results/tables/protocol_summary.md`

This generated report repeats the limitation that aggregate values are protocol
sanity outputs only.

## Deferred Work

- Do not implement Burgers1D for M6.
- Transformer integration is covered by M7.
- Do not add performance claims.

## Verification Commands

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/heat1d.yaml
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/heat1d_fno_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/heat1d_sm_fno_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/*_eval_metrics.json" --output results/tables/aggregate_metrics.json
PYTHONPATH=src python3 scripts/generate_protocol_report.py --aggregate results/tables/aggregate_metrics.json --output results/tables/protocol_summary.md
git diff --check
```

## Done Criteria

- Shared Heat1D comparison protocol is documented.
- Repeated-seed execution works for Heat1D FNO and SM-FNO smoke configs.
- Aggregation writes JSON and Markdown summaries.
- Protocol summary report generation works.
- Rollout-20 Heat1D configs exist for MLP, FNO, and SM-FNO.
- Tests cover toy metric aggregation and repeated-seed config expansion.
- Generated datasets, predictions, checkpoints, metrics, and reports remain
  ignored by git.
- M6 decision log and verification report document implementation decisions,
  commands run, artifact paths, and limitations.
