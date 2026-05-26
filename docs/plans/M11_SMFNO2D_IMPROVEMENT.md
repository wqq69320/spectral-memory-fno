# M11 SM-FNO2D Improvement Plan

## Objective

Harden the 2D spectral-memory path after the M10 Navier-Stokes2D protocol
extension. M11 addresses a reviewed FNO2D spectral-mode slicing bug, adds a
limited stabilized SM-FNO2D v2 variant, and verifies that the existing
config-driven smoke, rollout, repeated-seed, aggregate, and cost-efficiency
workflow can report the v2 model as a separate group.

M11 outputs are protocol-scale sanity artifacts only. They do not support
benchmark claims, cost-efficiency claims, or model rankings.

## Scope

- Fix `SpectralConv2D` retained-mode handling so small or odd 2D grids do not
  overlap top and bottom y-frequency writes.
- Add focused regression coverage for small-grid mode retention and gradient
  flow.
- Add a stable gated temporal memory block for SM-FNO2D v2.
- Add `SpectralMemoryFNO2DV2` with residual spatial spectral blocks and
  config-driven model construction.
- Add Navier-Stokes2D SM-FNO2D v2 smoke and rollout-20 protocol-validation
  configs.
- Add an M11 Navier-Stokes2D cost-efficiency analysis config that reports
  FNO2D, SM-FNO2D v1, SM-FNO2D v2, and Transformer2D as separate model groups.
- Verify direct smoke, rollout-20, repeated-seed, aggregate, horizon-sweep,
  generated-report, artifact-ignore, and whitespace checks.

## Done Criteria

- [x] `SpectralConv2D` clamps retained top and bottom y modes to non-overlapping
  frequency rows.
- [x] Regression tests cover small-grid FNO2D mode retention and used-weight
  gradients.
- [x] Stable gated diagonal SSM memory is available and covered by shape/range
  tests.
- [x] `SpectralMemoryFNO2DV2` is available through model exports and
  `scripts/train.py`.
- [x] Navier-Stokes2D SM-FNO2D v2 smoke and rollout-20 configs exist.
- [x] M11 cost-efficiency analysis includes `sm_fno2d_v2` as a distinct model
  group.
- [x] Full verification commands pass on current code.
- [x] Generated M11 data, checkpoints, metrics, predictions, expanded configs,
  figures, and generated reports are confirmed git-ignored.
- [x] `docs/reports/M11_VERIFICATION_REPORT.md` records commands, artifacts,
  smoke metric values, limitations, and no-claim guardrails.

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
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_v2_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_v2_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_v2_smoke.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_fno_rollout20.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_fno_rollout20.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_fno_rollout20.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_rollout20.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_rollout20.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_rollout20.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_v2_rollout20.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_v2_rollout20.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_v2_rollout20.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_transformer_rollout20.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_transformer_rollout20.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_transformer_rollout20.yaml
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_sm_fno_v2_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/navier_stokes2d*_eval_metrics.json" --output results/tables/navier_stokes2d_m11_aggregate_metrics.json
PYTHONPATH=src python3 scripts/run_horizon_sweep.py --config configs/analysis/m11_navier_stokes2d_v2_cost_smoke.yaml
PYTHONPATH=src python3 scripts/aggregate_cost_metrics.py --input-glob "results/tables/m11_navier_stokes2d_v2_cost/*_eval_metrics.json" --output results/tables/m11_navier_stokes2d_v2_cost_aggregate.json
PYTHONPATH=src python3 scripts/plot_cost_efficiency.py --aggregate results/tables/m11_navier_stokes2d_v2_cost_aggregate.json --output-dir results/figures/m11_navier_stokes2d_v2_cost
PYTHONPATH=src python3 scripts/generate_cost_efficiency_report.py --aggregate results/tables/m11_navier_stokes2d_v2_cost_aggregate.json --output results/tables/m11_navier_stokes2d_v2_cost_report.md
git diff --check
```

## Non-Goals

- Do not tune final 2D hyperparameters.
- Do not add production CFD validation.
- Do not change the M10 smoke protocol except by adding separate v2 configs.
- Do not claim that any model outperforms another model from M11 smoke,
  repeated-seed, rollout, aggregate, or cost-efficiency outputs.
