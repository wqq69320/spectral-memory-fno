# M10 Navier-Stokes2D Extension Plan

## Objective

Extend the Heat1D/Burgers1D fair-comparison and cost-efficiency workflow to
small 2D periodic Navier-Stokes vorticity forecasting. Preserve the 2D shape
contract:

`[batch, time, height, width, channels] -> [batch, pred_time, height, width, channels]`

M10 outputs are protocol-scale sanity artifacts only. They do not support
benchmark claims or model rankings.

## Scope

- Replace the Navier-Stokes placeholder with a CPU-friendly 2D periodic
  vorticity generator.
- Keep `configs/data/navier_stokes2d.yaml` small enough for local CPU smoke
  verification.
- Add 2D model configs and implementations for FNO2D, SM-FNO2D, and
  Transformer2D.
- Extend train/evaluate/rollout/plot paths to support 2D tensors.
- Add Navier-Stokes smoke and rollout-20 configs for all three 2D models.
- Add a Navier-Stokes horizon-sweep analysis config for the existing M9
  cost-efficiency runner.
- Add tests for 2D data generation, model shapes, rollout shape handling, and
  protocol config consistency.
- Verify data generation, smoke train/evaluate/plot, repeated-seed execution,
  aggregation, and artifact ignore behavior.

## Done Criteria

- [x] `generate_navier_stokes2d` returns finite tensors with shape
  `[samples, time, height, width, channels]`.
- [x] `scripts/generate_data.py` supports `name: navier_stokes2d`.
- [x] `PDEDataset` exposes generic spatial shape metadata.
- [x] `autoregressive_rollout` supports 5D input windows.
- [x] `FNO2D`, `SpectralMemoryFNO2D`, and `Transformer2DBaseline` are available
  through the model registry and `scripts/train.py`.
- [x] Navier-Stokes smoke configs exist for FNO2D, SM-FNO2D, and Transformer2D.
- [x] Navier-Stokes rollout-20 protocol-validation configs exist for FNO2D,
  SM-FNO2D, and Transformer2D.
- [x] Plotting handles 2D prediction snapshots.
- [x] Tests cover 2D generator, 2D model shapes, 2D rollout, and config
  consistency.
- [x] Full verification commands pass on current code.
- [x] Generated Navier-Stokes artifacts are confirmed git-ignored.
- [x] `docs/reports/M10_VERIFICATION_REPORT.md` records commands, artifacts,
  metric values as sanity checks only, limitations, and remaining risks.

## Required Verification

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/navier_stokes2d.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/navier_stokes2d*_eval_metrics.json" --output results/tables/navier_stokes2d_aggregate_metrics.json
PYTHONPATH=src python3 scripts/run_horizon_sweep.py --config configs/analysis/m10_navier_stokes2d_cost_smoke.yaml
PYTHONPATH=src python3 scripts/aggregate_cost_metrics.py --input-glob "results/tables/m10_navier_stokes2d_cost/*_eval_metrics.json" --output results/tables/m10_navier_stokes2d_cost_aggregate.json
git diff --check
```

## Non-Goals

- Do not implement production CFD validation.
- Do not tune 2D model hyperparameters.
- Do not add final benchmark-scale datasets.
- Do not claim model superiority from M10 smoke, repeated-seed, rollout, or
  cost-efficiency outputs.
