# M2-M5 Verification Report

Metric values in this report are smoke-run sanity checks only. They are not
benchmark claims and should not be used to rank models.

## Commands Run

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q` | Pass | 15 tests passed. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/heat1d.yaml` | Pass | Wrote Heat1D trajectories with shape `(128, 50, 64, 1)`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_mlp_smoke.yaml` | Pass | Wrote MLP checkpoint and train metrics under the shared smoke protocol. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_mlp_smoke.yaml` | Pass | Wrote MLP eval metrics, 5-step rollout metrics, and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_mlp_smoke.yaml` | Pass | Wrote MLP training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_fno_smoke.yaml` | Pass | Wrote FNO checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_fno_smoke.yaml` | Pass | Wrote FNO eval metrics and sample predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_fno_smoke.yaml` | Pass | Wrote FNO training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO eval metrics and sample predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO training, prediction, and rollout figures. |
| `git diff --check` | Pass | No whitespace errors. |

## Output Artifacts

- Dataset: `data/processed/heat1d/heat1d.npz`
- MLP checkpoint: `results/checkpoints/heat1d_mlp_smoke.pt`
- MLP train metrics: `results/tables/heat1d_mlp_smoke_train_metrics.json`
- MLP eval metrics: `results/tables/heat1d_mlp_smoke_eval_metrics.json`
- MLP predictions: `outputs/heat1d_mlp_smoke/predictions.npz`
- MLP figures:
  - `results/figures/heat1d_mlp_smoke_training_loss.png`
  - `results/figures/heat1d_mlp_smoke_prediction.png`
  - `results/figures/heat1d_mlp_smoke_rollout_relative_l2.png`
- FNO checkpoint: `results/checkpoints/heat1d_fno_smoke.pt`
- FNO train metrics: `results/tables/heat1d_fno_smoke_train_metrics.json`
- FNO eval metrics: `results/tables/heat1d_fno_smoke_eval_metrics.json`
- FNO predictions: `outputs/heat1d_fno_smoke/predictions.npz`
- FNO figures:
  - `results/figures/heat1d_fno_smoke_training_loss.png`
  - `results/figures/heat1d_fno_smoke_prediction.png`
  - `results/figures/heat1d_fno_smoke_rollout_relative_l2.png`
- SM-FNO checkpoint: `results/checkpoints/heat1d_sm_fno_smoke.pt`
- SM-FNO train metrics:
  `results/tables/heat1d_sm_fno_smoke_train_metrics.json`
- SM-FNO eval metrics:
  `results/tables/heat1d_sm_fno_smoke_eval_metrics.json`
- SM-FNO predictions: `outputs/heat1d_sm_fno_smoke/predictions.npz`
- SM-FNO figures:
  - `results/figures/heat1d_sm_fno_smoke_training_loss.png`
  - `results/figures/heat1d_sm_fno_smoke_prediction.png`
  - `results/figures/heat1d_sm_fno_smoke_rollout_relative_l2.png`

Generated datasets, checkpoints, metrics, predictions, and figures remain
git-ignored local artifacts.

## Metric Values

These values confirm that the pipeline produced finite outputs on the smoke
dataset using the shared Heat1D smoke protocol. They are not model rankings or
performance claims.

| Experiment | MSE | Relative L2 | Rollout Relative L2 | Rollout Steps |
| --- | ---: | ---: | ---: | ---: |
| `heat1d_mlp_smoke` | 0.0033383888 | 0.0947022376 | 0.1052094176 | 5 |
| `heat1d_fno_smoke` | 0.0003802054 | 0.0324720088 | 0.0377987958 | 5 |
| `heat1d_sm_fno_smoke` | 0.0005952498 | 0.0414182431 | 0.0954043940 | 5 |

Per-timestep rollout relative L2 values were written to each evaluation metrics
JSON under `rollout_relative_l2_per_timestep`.

## Pass/Fail Summary

The M2-M5 Research MVP Done Criteria are satisfied by the current smoke scope:

- Tests pass.
- Existing Heat1D MLP smoke train/evaluate/plot path works with the same
  5-step rollout horizon as the FNO and SM-FNO smoke configs.
- Heat1D FNO smoke train/evaluate/plot works.
- Heat1D SM-FNO smoke train/evaluate/plot works.
- Basic autoregressive rollout evaluation works.
- Per-timestep rollout relative L2 and inference timing are logged.
- README, plan, decision log, and verification report are updated without
  benchmark claims.

## Remaining Risks

- Smoke experiments use a small synthetic Heat1D dataset and short rollout
  horizon, so they verify pipeline mechanics only.
- Single-seed smoke metrics do not establish robustness.
- Inference timing is wall-clock timing on small CPU batches and is not a
  benchmark.
- The diagonal SSM is intentionally simple; stronger temporal baselines remain
  future work.
