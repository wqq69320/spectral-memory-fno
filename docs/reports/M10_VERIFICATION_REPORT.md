# M10 Verification Report

M10 outputs are protocol-scale sanity checks only. They are not final benchmark
claims, they do not rank models, and they should not be used to claim that one
model is more accurate or cost-efficient than another.

## Verification Status

Verification passed on the current code. Generated Navier-Stokes artifacts
remain under ignored local output paths.

## Commands

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q tests/test_shapes.py tests/test_navier_stokes2d_data.py tests/test_imports.py` | Pass | 17 focused 2D/data/import tests passed. |
| `PYTHONPATH=src pytest -q` | Pass | 34 tests passed after the final code changes. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/navier_stokes2d.yaml` | Pass | Wrote `data/processed/navier_stokes2d/navier_stokes2d.npz` with shape `(64, 36, 16, 16, 1)`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml` | Pass | Wrote FNO2D smoke checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml` | Pass | Wrote FNO2D smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml` | Pass | Wrote FNO2D smoke training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO2D smoke checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO2D smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO2D smoke training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml` | Pass | Wrote Transformer2D smoke checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml` | Pass | Wrote Transformer2D smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_transformer_smoke.yaml` | Pass | Wrote Transformer2D smoke training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_fno_smoke.yaml --seeds 0 1` | Pass | Wrote FNO2D repeated-seed checkpoints, metrics, and predictions. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_sm_fno_smoke.yaml --seeds 0 1` | Pass | Wrote SM-FNO2D repeated-seed checkpoints, metrics, and predictions. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_fno_rollout20.yaml` | Pass | Rollout-20 FNO2D config ran. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_fno_rollout20.yaml` | Pass | Wrote rollout-20 FNO2D eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_fno_rollout20.yaml` | Pass | Wrote rollout-20 FNO2D figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_sm_fno_rollout20.yaml` | Pass | Rollout-20 SM-FNO2D config ran. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_sm_fno_rollout20.yaml` | Pass | Wrote rollout-20 SM-FNO2D eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_sm_fno_rollout20.yaml` | Pass | Wrote rollout-20 SM-FNO2D figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_transformer_rollout20.yaml` | Pass | Rollout-20 Transformer2D config ran. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_transformer_rollout20.yaml` | Pass | Wrote rollout-20 Transformer2D eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_transformer_rollout20.yaml` | Pass | Wrote rollout-20 Transformer2D figures. |
| `PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/navier_stokes2d*_eval_metrics.json" --output results/tables/navier_stokes2d_aggregate_metrics.json` | Pass | Aggregated 10 Navier-Stokes eval metric files separately. |
| `PYTHONPATH=src python3 scripts/run_horizon_sweep.py --config configs/analysis/m10_navier_stokes2d_cost_smoke.yaml` | Pass | Ran 3 models x 2 seeds x horizons 5/10/20; wrote 24 expanded configs. |
| `PYTHONPATH=src python3 scripts/aggregate_cost_metrics.py --input-glob "results/tables/m10_navier_stokes2d_cost/*_eval_metrics.json" --output results/tables/m10_navier_stokes2d_cost_aggregate.json` | Pass | Aggregated 18 M10 cost eval files into 9 dataset/model/horizon groups. |
| `PYTHONPATH=src python3 scripts/plot_cost_efficiency.py --aggregate results/tables/m10_navier_stokes2d_cost_aggregate.json --output-dir results/figures/m10_navier_stokes2d_cost` | Pass | Wrote 3 M10 cost-efficiency plots. |
| `PYTHONPATH=src python3 scripts/generate_cost_efficiency_report.py --aggregate results/tables/m10_navier_stokes2d_cost_aggregate.json --output results/tables/m10_navier_stokes2d_cost_report.md` | Pass | Wrote generated M10 cost-efficiency report. |
| `git diff --check` | Pass | No whitespace errors. |
| `git check-ignore -v ...` | Pass | Confirmed representative generated Navier-Stokes data, checkpoints, metrics, predictions, expanded configs, reports, and figures are ignored. |

## Output Artifacts

Generated artifacts:

- `data/processed/navier_stokes2d/navier_stokes2d.npz`
- `results/checkpoints/navier_stokes2d_*.pt`
- `results/tables/navier_stokes2d_*_train_metrics.json`
- `results/tables/navier_stokes2d_*_eval_metrics.json` (10 direct/repeated/rollout eval files)
- `outputs/navier_stokes2d_*/predictions.npz`
- `results/figures/navier_stokes2d_*.png` (18 direct smoke/rollout figures)
- `outputs/repeated_seeds/navier_stokes2d_*/seed_*/predictions.npz`
- `results/tables/navier_stokes2d_aggregate_metrics.json`
- `results/tables/navier_stokes2d_aggregate_metrics.md`
- `outputs/m10_navier_stokes2d_cost_expanded_configs/*.yaml` (24 expanded configs)
- `results/tables/m10_navier_stokes2d_cost/*_metrics.json`
- `results/tables/m10_navier_stokes2d_cost_aggregate.json`
- `results/tables/m10_navier_stokes2d_cost_aggregate.md`
- `results/tables/m10_navier_stokes2d_cost_report.md`
- `results/figures/m10_navier_stokes2d_cost/*.png` (3 cost plots)

These paths were checked as local ignored artifacts.

## Aggregate Coverage

- Direct/repeated/rollout aggregate: FNO2D, SM-FNO2D, and Transformer2D smoke
  runs; FNO2D and SM-FNO2D repeated seeds `0`, `1`; all three rollout-20
  configs.
- Cost aggregate: FNO2D, SM-FNO2D, and Transformer2D at horizons `5`, `10`,
  `20`, with seeds `0`, `1`.

Metric values are recorded as smoke and protocol-validation sanity checks only.

## Metrics Recorded

- One-step MSE and relative L2.
- Rollout MSE, rollout relative L2, and per-timestep rollout relative L2.
- Inference timing fields from evaluation.
- Parameter-count fields from train/evaluate metrics.

## Remaining Risks

- The Navier-Stokes generator is a small pseudo-spectral protocol fixture.
- The dataset is smooth, low resolution, and CPU-friendly by design.
- Two repeated seeds verify workflow mechanics only.
- Rollout-20 and horizon-sweep outputs validate longer-horizon plumbing, not
  final benchmark behavior.
