# M11 Verification Report

M11 outputs are protocol-scale sanity checks only. They are not final benchmark
claims, they do not rank models, and they should not be used to claim that one
model is more accurate or cost-efficient than another.

## Verification Status

Verification passed on the current code. The FNO2D small-grid spectral-mode
review finding is resolved by clamping retained top and bottom y-frequency
bands to non-overlapping rows. Generated M11 artifacts remain under ignored
local output paths.

## Commands

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q tests/test_shapes.py::test_spectral_conv2d_small_grid_modes_do_not_overlap` | Pass | Focused regression for non-overlapping 8x8 and odd 15x15 retained y modes passed. |
| `PYTHONPATH=src pytest -q tests/test_shapes.py tests/test_imports.py` | Pass | 17 focused shape/import tests passed after adding SM-FNO2D v2. |
| `PYTHONPATH=src pytest -q tests/test_shapes.py tests/test_imports.py tests/test_navier_stokes2d_data.py tests/test_m9_cost_efficiency.py` | Pass | 24 focused shape/import/config/cost tests passed. |
| `PYTHONPATH=src pytest -q` | Pass | 37 tests passed. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/navier_stokes2d.yaml` | Pass | Wrote `data/processed/navier_stokes2d/navier_stokes2d.npz` with shape `(64, 36, 16, 16, 1)`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml` | Pass | Wrote FNO2D smoke checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml` | Pass | Wrote FNO2D smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml` | Pass | Wrote FNO2D smoke plots. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO2D v1 smoke checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO2D v1 smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO2D v1 smoke plots. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_v2_smoke.yaml` | Pass | Wrote SM-FNO2D v2 smoke checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_v2_smoke.yaml` | Pass | Wrote SM-FNO2D v2 smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_v2_smoke.yaml` | Pass | Wrote SM-FNO2D v2 smoke plots. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml` | Pass | Wrote Transformer2D smoke checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml` | Pass | Wrote Transformer2D smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml` | Pass | Wrote Transformer2D smoke plots. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_fno_rollout20.yaml` | Pass | Rollout-20 FNO2D config ran. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_fno_rollout20.yaml` | Pass | Wrote rollout-20 FNO2D eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_fno_rollout20.yaml` | Pass | Wrote rollout-20 FNO2D plots. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_rollout20.yaml` | Pass | Rollout-20 SM-FNO2D v1 config ran. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_rollout20.yaml` | Pass | Wrote rollout-20 SM-FNO2D v1 eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_rollout20.yaml` | Pass | Wrote rollout-20 SM-FNO2D v1 plots. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_v2_rollout20.yaml` | Pass | Rollout-20 SM-FNO2D v2 config ran. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_v2_rollout20.yaml` | Pass | Wrote rollout-20 SM-FNO2D v2 eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_v2_rollout20.yaml` | Pass | Wrote rollout-20 SM-FNO2D v2 plots. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_transformer_rollout20.yaml` | Pass | Rollout-20 Transformer2D config ran. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_transformer_rollout20.yaml` | Pass | Wrote rollout-20 Transformer2D eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_transformer_rollout20.yaml` | Pass | Wrote rollout-20 Transformer2D plots. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml --seeds 0 1` | Pass | Wrote FNO2D repeated-seed checkpoints, metrics, and predictions. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml --seeds 0 1` | Pass | Wrote SM-FNO2D v1 repeated-seed checkpoints, metrics, and predictions. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_sm_fno_v2_smoke.yaml --seeds 0 1` | Pass | Wrote SM-FNO2D v2 repeated-seed checkpoints, metrics, and predictions. |
| `PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/navier_stokes2d*_eval_metrics.json" --output results/tables/navier_stokes2d_m11_aggregate_metrics.json` | Pass | Aggregated direct, rollout-20, and repeated-seed Navier-Stokes2D eval files. |
| `PYTHONPATH=src python3 scripts/run_horizon_sweep.py --config configs/analysis/m11_navier_stokes2d_v2_cost_smoke.yaml` | Pass | Ran 4 models x 2 seeds x horizons 5/10/20 and wrote 32 expanded configs. |
| `PYTHONPATH=src python3 scripts/aggregate_cost_metrics.py --input-glob "results/tables/m11_navier_stokes2d_v2_cost/*_eval_metrics.json" --output results/tables/m11_navier_stokes2d_v2_cost_aggregate.json` | Pass | Aggregated 24 M11 cost eval files into 12 dataset/model/horizon groups. |
| `PYTHONPATH=src python3 scripts/plot_cost_efficiency.py --aggregate results/tables/m11_navier_stokes2d_v2_cost_aggregate.json --output-dir results/figures/m11_navier_stokes2d_v2_cost` | Pass | Wrote 3 M11 cost-efficiency plots. |
| `PYTHONPATH=src python3 scripts/generate_cost_efficiency_report.py --aggregate results/tables/m11_navier_stokes2d_v2_cost_aggregate.json --output results/tables/m11_navier_stokes2d_v2_cost_report.md` | Pass | Wrote generated M11 cost-efficiency report. |
| `git diff --check` | Pass | No whitespace errors. |
| `git check-ignore -v ...` | Pass | Confirmed representative generated M11 data, checkpoints, metrics, predictions, expanded configs, reports, and figures are ignored. |

## Output Artifacts

Generated artifacts:

- `data/processed/navier_stokes2d/navier_stokes2d.npz`
- `results/checkpoints/navier_stokes2d_*.pt`
- `results/tables/navier_stokes2d_*_train_metrics.json`
- `results/tables/navier_stokes2d_*_eval_metrics.json`
- `outputs/navier_stokes2d_*/predictions.npz`
- `results/figures/navier_stokes2d_*.png`
- `outputs/repeated_seeds/navier_stokes2d_*/seed_*/predictions.npz`
- `results/tables/navier_stokes2d_m11_aggregate_metrics.json`
- `results/tables/navier_stokes2d_m11_aggregate_metrics.md`
- `outputs/m11_navier_stokes2d_v2_cost_expanded_configs/*.yaml`
- `results/tables/m11_navier_stokes2d_v2_cost/*_metrics.json`
- `outputs/m11_navier_stokes2d_v2_cost/*/seed_*/horizon_*/predictions.npz`
- `results/tables/m11_navier_stokes2d_v2_cost_aggregate.json`
- `results/tables/m11_navier_stokes2d_v2_cost_aggregate.md`
- `results/tables/m11_navier_stokes2d_v2_cost_report.md`
- `results/figures/m11_navier_stokes2d_v2_cost/*.png`

These paths were checked as local ignored artifacts.

## Metrics Recorded

Direct smoke and rollout-20 outputs:

| Experiment | MSE | Relative L2 | Rollout Relative L2 |
| --- | ---: | ---: | ---: |
| `navier_stokes2d_fno_smoke` | 0.011339 | 0.251409 | 0.286220 |
| `navier_stokes2d_sm_fno_smoke` | 0.071001 | 0.630850 | 0.915623 |
| `navier_stokes2d_sm_fno_v2_smoke` | 0.009724 | 0.233165 | 0.431524 |
| `navier_stokes2d_transformer_smoke` | 0.001639 | 0.094142 | 0.128788 |
| `navier_stokes2d_fno_rollout20` | 0.011339 | 0.251409 | 0.447127 |
| `navier_stokes2d_sm_fno_rollout20` | 0.071001 | 0.630850 | 1.030457 |
| `navier_stokes2d_sm_fno_v2_rollout20` | 0.009724 | 0.233165 | 0.658295 |
| `navier_stokes2d_transformer_rollout20` | 0.001639 | 0.094142 | 0.222797 |

Repeated-seed rollout relative L2 values:

| Experiment | Seeds | Rollout Relative L2 Values |
| --- | --- | --- |
| `navier_stokes2d_fno_smoke` | `0, 1` | 0.293882, 0.559482 |
| `navier_stokes2d_sm_fno_smoke` | `0, 1` | 0.992293, 0.793398 |
| `navier_stokes2d_sm_fno_v2_smoke` | `0, 1` | 0.319764, 0.458169 |

The recorded protocol-scale outputs show that SM-FNO2D v2 reduced SM-FNO2D v1
sanity errors in this small run, while also adding local CPU time. This is a
useful implementation signal only; it is not benchmark evidence or a model
ranking.

## Aggregate Coverage

- Direct/repeated/rollout aggregate:
  FNO2D, SM-FNO2D v1, SM-FNO2D v2, and Transformer2D smoke runs; FNO2D,
  SM-FNO2D v1, and SM-FNO2D v2 repeated seeds `0`, `1`; all four rollout-20
  configs.
- Cost aggregate:
  FNO2D, SM-FNO2D v1, SM-FNO2D v2, and Transformer2D at horizons `5`, `10`,
  `20`, with seeds `0`, `1`.

Metric values are recorded as smoke and protocol-validation sanity checks only.

## Remaining Risks

- SM-FNO2D v2 is a limited stabilization path, not a tuned final architecture.
- The Navier-Stokes2D dataset is smooth, low resolution, and CPU-friendly by
  design.
- Two repeated seeds verify workflow mechanics only.
- Timing values are local wall-clock measurements and can vary by machine load.
- Rollout-20 and horizon-sweep outputs validate longer-horizon plumbing, not
  final benchmark behavior.
