# M16 Dynamic Navier-Stokes2D Report

M16 outputs are diagnostic artifacts only. They are not benchmark claims, model
rankings, or evidence of general PDE forecasting performance.

## Objective

M15 showed that the medium Navier-Stokes2D fixture changed slowly and that
SM-FNO2D v3 did not beat last-frame persistence. M16 creates a more dynamic
CPU-feasible fixture and reruns FNO2D, SM-FNO2D v3, and persistence diagnostics
under a 36-step rollout.

## Dynamic Fixture

The dynamic fixture is:

- config: `configs/data/navier_stokes2d_dynamic.yaml`
- output: `data/processed/navier_stokes2d_dynamic/navier_stokes2d_dynamic.npz`
- shape: `(36, 76, 24, 24, 1)`
- internal `dt`: `0.004`
- `save_every`: `10`
- effective saved-frame spacing: `0.04`
- viscosity: `0.0005`
- initial amplitude: `1.25`
- max initial-condition modes: `5`
- seed: `52`

The solver still checks the explicit advection CFL condition at every internal
step. Data generation completed without CFL or non-finite-value failures.

## Commands Run

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q tests/test_navier_stokes2d_data.py tests/test_dynamic_skill.py` | Pass | 11 focused tests passed after generator/config changes. |
| `python3 -m py_compile src/sm_fno/data/navier_stokes2d.py scripts/generate_data.py scripts/run_dynamic_skill_diagnostics.py` | Pass | Edited generator and diagnostic scripts compile. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/navier_stokes2d_dynamic.yaml` | Pass | Wrote dynamic dataset with shape `(36, 76, 24, 24, 1)`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml` | Pass | Wrote FNO2D checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml` | Pass | Wrote FNO2D eval metrics and prediction artifact. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml` | Pass | Wrote FNO2D training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_diagnostic.yaml` | Pass | Wrote SM-FNO2D v3 checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_diagnostic.yaml` | Pass | Wrote SM-FNO2D v3 eval metrics and prediction artifact. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_diagnostic.yaml` | Pass | Wrote SM-FNO2D v3 training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/run_dynamic_skill_diagnostics.py --experiment-config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml --experiment-config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_diagnostic.yaml --output-json results/tables/m16_dynamic_ns2d_skill_diagnostics.json --output-md results/tables/m16_dynamic_ns2d_skill_diagnostics.md --figures-dir results/figures/m16_dynamic_ns2d --artifact-prefix m16_dynamic_ns2d --title-prefix M16 --v3-experiment navier_stokes2d_dynamic_sm_fno_v3_diagnostic` | Pass | Wrote dynamic-skill JSON/Markdown, comparison plots, and v3 step-36 surface. |
| `python3 - <<'PY' ... inspect M16 JSON and image dimensions ... PY` | Pass | Confirmed metrics and generated figure dimensions. |
| `PYTHONPATH=src pytest -q` | Pass | 51 tests passed after M16 changes. |
| `python3 -m py_compile src/sm_fno/data/navier_stokes2d.py scripts/generate_data.py scripts/run_dynamic_skill_diagnostics.py` | Pass | Final edited generator and diagnostic scripts compile. |
| `git diff --check` | Pass | No whitespace errors. |
| `git check-ignore -v ... representative M16 artifacts ...` | Pass | Confirmed generated dynamic data, predictions, checkpoints, tables, and figures are ignored. |

## Metrics

The new fixture is persistence-hard relative to M15. Held-out accumulated
change from the last input frame is no longer near-zero:

| Held-Out Diagnostic | Mean | Step 36 |
| --- | ---: | ---: |
| Persistence error / change from last input | 0.127381 | 0.238863 |
| True step-to-step change | 0.006286 | 0.005398 |

Model rollout diagnostics:

| Model | One-Step Rel L2 | Rollout-36 Rel L2 | Rollout Spectrum Error | Step 1 Rel L2 | Step 36 Rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FNO2D | 0.045675 | 0.145452 | 0.119530 | 0.054440 | 0.226624 |
| SM-FNO2D v3 | 0.047404 | 0.149982 | 0.125461 | 0.053792 | 0.238711 |

Persistence-normalized dynamic skill from saved sample rollout artifacts:

| Model | Model Rel L2 Mean | Model Rel L2 Step 36 | Persistence Rel L2 Mean | Persistence Rel L2 Step 36 | Skill Mean | Skill Step 36 | Delta Error Mean | Delta Error Step 36 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FNO2D | 0.142422 | 0.228615 | 0.145042 | 0.272864 | -0.323437 | 0.162162 | 1.323437 | 0.837838 |
| SM-FNO2D v3 | 0.147972 | 0.245102 | 0.145042 | 0.272864 | -0.348548 | 0.101741 | 1.348548 | 0.898259 |

## Interpretation

The M16 fixture is meaningfully more dynamic than the M15 medium fixture for
36-step persistence-hard rollout diagnostics. The held-out persistence error is
`0.127381` on average and `0.238863` at step 36, so last-frame persistence is no
longer a near-zero-error baseline over the horizon.

The fixture is still not strongly chaotic at the saved-step level: true
step-to-step relative change is about `0.006286` on average. This means the
fixture is useful for testing whether models beat persistence over accumulated
rollout change, but it is not a high-dynamics CFD benchmark.

On the saved sample artifacts, both FNO2D and SM-FNO2D v3 have positive
step-36 skill versus persistence, but negative mean skill over the full
36-step horizon. SM-FNO2D v3 therefore is **not meaningfully better than
persistence overall** under the M16 diagnostic criterion, even though it is in
the same rollout-error range as FNO2D and no v2-style collapse is observed.

## Output Artifacts

Data and model artifacts:

- `data/processed/navier_stokes2d_dynamic/navier_stokes2d_dynamic.npz`
- `results/checkpoints/navier_stokes2d_dynamic_fno_diagnostic.pt`
- `results/checkpoints/navier_stokes2d_dynamic_sm_fno_v3_diagnostic.pt`
- `outputs/navier_stokes2d_dynamic_fno_diagnostic/predictions.npz`
- `outputs/navier_stokes2d_dynamic_sm_fno_v3_diagnostic/predictions.npz`

Metric artifacts:

- `results/tables/navier_stokes2d_dynamic_fno_diagnostic_train_metrics.json`
- `results/tables/navier_stokes2d_dynamic_fno_diagnostic_eval_metrics.json`
- `results/tables/navier_stokes2d_dynamic_sm_fno_v3_diagnostic_train_metrics.json`
- `results/tables/navier_stokes2d_dynamic_sm_fno_v3_diagnostic_eval_metrics.json`
- `results/tables/m16_dynamic_ns2d_skill_diagnostics.json`
- `results/tables/m16_dynamic_ns2d_skill_diagnostics.md`

Figures:

- `results/figures/navier_stokes2d_dynamic_fno_diagnostic_training_loss.png`
- `results/figures/navier_stokes2d_dynamic_fno_diagnostic_prediction.png`
- `results/figures/navier_stokes2d_dynamic_fno_diagnostic_rollout_relative_l2.png`
- `results/figures/navier_stokes2d_dynamic_sm_fno_v3_diagnostic_training_loss.png`
- `results/figures/navier_stokes2d_dynamic_sm_fno_v3_diagnostic_prediction.png`
- `results/figures/navier_stokes2d_dynamic_sm_fno_v3_diagnostic_rollout_relative_l2.png`
- `results/figures/m16_dynamic_ns2d/m16_dynamic_ns2d_rollout_vs_persistence.png`
- `results/figures/m16_dynamic_ns2d/m16_dynamic_ns2d_heldout_temporal_variation.png`
- `results/figures/m16_dynamic_ns2d/m16_dynamic_ns2d_persistence_skill.png`
- `results/figures/m16_dynamic_ns2d/m16_dynamic_ns2d_delta_prediction_error.png`
- `results/figures/m16_dynamic_ns2d/m16_dynamic_ns2d_sm_fno2d_v3_surface_step36.png`

Generated M16 dynamic-skill figure dimensions:

- Rollout vs persistence: `1440 x 864`
- Held-out temporal variation: `1440 x 864`
- Persistence skill: `1440 x 864`
- Delta-prediction error: `1440 x 864`
- SM-FNO2D v3 step-36 surface: `1533 x 522`

These are generated artifacts under ignored local output paths.

## Limitations

- Single seed only.
- 24x24 synthetic vorticity fixture only.
- The dynamic fixture is more persistence-hard over 36 steps, but per-step
  temporal change is still modest.
- Model-specific persistence skill uses saved single-sample rollout artifacts.
- Hyperparameters are smoke-diagnostic settings, not tuned training budgets.
- No benchmark claims should be made from M16.

## Remaining Risks

- A future milestone should save full test-set rollout predictions for
  persistence skill across all held-out trajectories.
- More dynamic regimes may require forcing, larger grids, or different
  viscosity/amplitude settings.
- Stability and skill conclusions may change across seeds and training budgets.
