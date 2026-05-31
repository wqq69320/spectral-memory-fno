# M12 Medium Navier-Stokes2D Diagnostic Report

This report documents a medium-grid long-horizon Navier-Stokes2D diagnostic.
It is not a final benchmark, does not rank models, and should not be used to
claim that one model is more accurate or cost-efficient than another.

## Verification Status

The medium diagnostic protocol completed on the current code:

- Dataset: 24x24 periodic Navier-Stokes2D vorticity fixture.
- Samples: 36 trajectories.
- Time steps: 76.
- Input steps: 10.
- Prediction steps: 1.
- Rollout steps: 36.
- Models completed: FNO2D, SM-FNO2D v2, Transformer2D.
- Repeated seeds completed: FNO2D and SM-FNO2D v2 with seeds `0`, `1`.
- Transformer2D repeated seeds were deferred to keep this diagnostic bounded;
  the matched single-seed Transformer2D train/evaluate/plot path completed.

The protocol remained stable for the medium fixture: data generation,
training, evaluation, plotting, aggregate metrics, cost summaries, tests, and
artifact-ignore checks completed.

## Commands

| Command | Status | Runtime | Notes |
| --- | --- | ---: | --- |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/navier_stokes2d_medium.yaml` | Pass | 4.78s | Wrote the medium dataset with shape `(36, 76, 24, 24, 1)`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml` | Pass | 3.97s | Wrote FNO2D checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml` | Pass | 1.00s | Wrote FNO2D eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml` | Pass | 1.75s | Wrote FNO2D training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml` | Pass | 21.05s | Wrote SM-FNO2D v2 checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml` | Pass | 1.95s | Wrote SM-FNO2D v2 eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml` | Pass | 1.40s | Wrote SM-FNO2D v2 training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_medium_transformer_diagnostic.yaml` | Pass | 25.04s | Wrote Transformer2D checkpoint and training metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_medium_transformer_diagnostic.yaml` | Pass | 4.43s | Wrote Transformer2D eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_medium_transformer_diagnostic.yaml` | Pass | 1.43s | Wrote Transformer2D training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml --seeds 0 1` | Pass | 8.56s | Wrote FNO2D repeated-seed train/eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml --seeds 0 1` | Pass | 48.25s | Wrote SM-FNO2D v2 repeated-seed train/eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/navier_stokes2d_medium_*_diagnostic*_eval_metrics.json" --output results/tables/m12_navier_stokes2d_medium_aggregate_metrics.json` | Pass | <1s | Wrote medium single/repeated aggregate metrics. |
| `PYTHONPATH=src python3 scripts/aggregate_cost_metrics.py --input-glob "results/tables/navier_stokes2d_medium_*_diagnostic_eval_metrics.json" --output results/tables/m12_navier_stokes2d_medium_cost_aggregate.json` | Pass | 0.09s | Wrote matched single-seed cost aggregate. |
| `PYTHONPATH=src python3 scripts/plot_cost_efficiency.py --aggregate results/tables/m12_navier_stokes2d_medium_cost_aggregate.json --output-dir results/figures/m12_navier_stokes2d_medium_cost` | Pass | 0.64s | Wrote 3 medium cost-summary plots. |
| `PYTHONPATH=src python3 scripts/generate_cost_efficiency_report.py --aggregate results/tables/m12_navier_stokes2d_medium_cost_aggregate.json --output results/tables/m12_navier_stokes2d_medium_cost_report.md` | Pass | 0.04s | Wrote generated medium cost-efficiency report. |
| `PYTHONPATH=src pytest -q` | Pass | 1.30s | 37 tests passed. |
| `git diff --check` | Pass | <1s | No whitespace errors. |
| `git check-ignore -v ...` | Pass | <1s | Representative generated data, checkpoints, metrics, predictions, and figures are ignored. |

## Output Artifacts

Generated local artifacts:

- `data/processed/navier_stokes2d_medium/navier_stokes2d_medium.npz`
- `results/checkpoints/navier_stokes2d_medium_fno_diagnostic.pt`
- `results/checkpoints/navier_stokes2d_medium_sm_fno_v2_diagnostic.pt`
- `results/checkpoints/navier_stokes2d_medium_transformer_diagnostic.pt`
- `results/checkpoints/navier_stokes2d_medium_fno_diagnostic_seed0.pt`
- `results/checkpoints/navier_stokes2d_medium_fno_diagnostic_seed1.pt`
- `results/checkpoints/navier_stokes2d_medium_sm_fno_v2_diagnostic_seed0.pt`
- `results/checkpoints/navier_stokes2d_medium_sm_fno_v2_diagnostic_seed1.pt`
- `results/tables/navier_stokes2d_medium_*_diagnostic_train_metrics.json`
- `results/tables/navier_stokes2d_medium_*_diagnostic_eval_metrics.json`
- `results/tables/navier_stokes2d_medium_*_diagnostic_seed*_train_metrics.json`
- `results/tables/navier_stokes2d_medium_*_diagnostic_seed*_eval_metrics.json`
- `outputs/navier_stokes2d_medium_fno_diagnostic/predictions.npz`
- `outputs/navier_stokes2d_medium_sm_fno_v2_diagnostic/predictions.npz`
- `outputs/navier_stokes2d_medium_transformer_diagnostic/predictions.npz`
- `outputs/repeated_seeds/navier_stokes2d_medium_fno_diagnostic/seed_*/predictions.npz`
- `outputs/repeated_seeds/navier_stokes2d_medium_sm_fno_v2_diagnostic/seed_*/predictions.npz`
- `results/figures/navier_stokes2d_medium_fno_diagnostic_training_loss.png`
- `results/figures/navier_stokes2d_medium_fno_diagnostic_prediction.png`
- `results/figures/navier_stokes2d_medium_fno_diagnostic_rollout_relative_l2.png`
- `results/figures/navier_stokes2d_medium_sm_fno_v2_diagnostic_training_loss.png`
- `results/figures/navier_stokes2d_medium_sm_fno_v2_diagnostic_prediction.png`
- `results/figures/navier_stokes2d_medium_sm_fno_v2_diagnostic_rollout_relative_l2.png`
- `results/figures/navier_stokes2d_medium_transformer_diagnostic_training_loss.png`
- `results/figures/navier_stokes2d_medium_transformer_diagnostic_prediction.png`
- `results/figures/navier_stokes2d_medium_transformer_diagnostic_rollout_relative_l2.png`
- `results/tables/m12_navier_stokes2d_medium_aggregate_metrics.json`
- `results/tables/m12_navier_stokes2d_medium_aggregate_metrics.md`
- `results/tables/m12_navier_stokes2d_medium_cost_aggregate.json`
- `results/tables/m12_navier_stokes2d_medium_cost_aggregate.md`
- `results/tables/m12_navier_stokes2d_medium_cost_report.md`
- `results/figures/m12_navier_stokes2d_medium_cost/rollout_relative_l2_by_horizon.png`
- `results/figures/m12_navier_stokes2d_medium_cost/rollout_seconds_per_step_by_horizon.png`
- `results/figures/m12_navier_stokes2d_medium_cost/rollout_error_vs_seconds_per_step.png`

These paths are generated artifacts and remain ignored by git.

## Metrics Recorded

Matched single-seed medium diagnostic metrics:

| Model | Seed | MSE | Relative L2 | Rollout Relative L2 | Rollout Seconds/Step |
| --- | ---: | ---: | ---: | ---: | ---: |
| FNO2D | 42 | 0.000101 | 0.024745 | 0.081641 | 0.001263 |
| SM-FNO2D v2 | 42 | 0.000081 | 0.021313 | 0.763503 | 0.011007 |
| Transformer2D | 42 | 0.000415 | 0.049907 | 0.231898 | 0.035924 |

Repeated-seed diagnostic metrics:

| Model | Seeds | Mean Rollout Relative L2 | Notes |
| --- | --- | ---: | --- |
| FNO2D | `0, 1` | 0.044350 +/- 0.015762 | Repeated-seed reporting check only. |
| SM-FNO2D v2 | `0, 1` | 1.409235 +/- 0.109671 | Repeated-seed reporting check only. |

These values are medium diagnostic outputs only. They are not benchmark
evidence, and the single-seed rows should not be interpreted as a model
ranking.

## Runtime Limitations

- The medium grid uses 24x24 rather than 32x32 to stay CPU-feasible.
- Training budgets are intentionally small: 2 epochs for each model.
- Repeated seeds were run only for FNO2D and SM-FNO2D v2. Transformer2D
  completed the matched single-seed train/evaluate/plot path, but repeated
  Transformer2D seeds were deferred because the single Transformer2D training
  and evaluation path was the slowest model path.
- Timing values are local CPU wall-clock measurements and can vary with system
  load.

## Stability Assessment

The 24x24, 76-step, rollout-36 diagnostic remained stable under the existing
protocol scripts. The data generator produced finite trajectories, all three
single-seed model paths trained and evaluated, rollout curves and prediction
snapshots were generated, and cost-summary artifacts were produced. This
supports the next step of larger diagnostic grids or stronger repeated-seed
coverage, but it does not support benchmark claims.

## Remaining Risks

- The Navier-Stokes2D data remains a low-resolution protocol fixture.
- No physical diagnostics such as energy or spectrum error are included.
- The repeated-seed coverage is partial.
- Cost summaries use local timing and parameter counts only.
- Hyperparameters are diagnostic settings, not tuned final settings.
