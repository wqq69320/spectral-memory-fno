# M14 Verification Report

M14 outputs are medium Navier-Stokes2D diagnostic checks only. They are not
final benchmark claims, they do not rank models, and they should not be used to
claim that one architecture is generally better than another.

## Verification Status

Verification passed on the current code. SM-FNO2D v3 trained, evaluated,
plotted, and produced 36-step rollout metrics under the medium protocol. In
this single-seed diagnostic, v3 fixed the observed SM-FNO2D v2 rollout
collapse while keeping one-step error in the same small diagnostic range.

## Implementation Summary

- Added `SpectralMemoryFNO2DV3` as a separate model.
- v3 predicts `FNO2D(inputs) + gate * SSM correction`.
- The SSM correction gate is initialized conservatively and capped by
  `gate_limit`.
- Added config-driven rollout-aware training through `rollout_train_steps` and
  `rollout_loss_weight`.
- Added one-step and rollout diagnostic stats:
  prediction stats, target stats, per-timestep rollout relative L2, and spatial
  Fourier spectrum error.

## Commands

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q tests/test_shapes.py tests/test_imports.py tests/test_navier_stokes2d_data.py tests/test_metrics.py` | Pass | 28 focused model/config/metric tests passed. |
| `python3 -m py_compile scripts/train.py scripts/evaluate.py src/sm_fno/models/sm_fno2d.py src/sm_fno/training/trainer.py src/sm_fno/evaluation/metrics.py` | Pass | Edited Python modules compile. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/navier_stokes2d_medium.yaml` | Pass | Wrote medium dataset with shape `(36, 76, 24, 24, 1)`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml` | Pass | Wrote FNO2D checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml` | Pass | Wrote FNO2D eval metrics and prediction artifact. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml` | Pass | Wrote FNO2D training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml` | Pass | Wrote SM-FNO2D v2 checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml` | Pass | Wrote SM-FNO2D v2 eval metrics and prediction artifact. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml` | Pass | Wrote SM-FNO2D v2 training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v3_diagnostic.yaml` | Pass | Wrote SM-FNO2D v3 checkpoint and rollout-aware train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v3_diagnostic.yaml` | Pass | Wrote SM-FNO2D v3 eval metrics and prediction artifact. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v3_diagnostic.yaml` | Pass | Wrote SM-FNO2D v3 training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/navier_stokes2d_medium_*fno*_diagnostic_eval_metrics.json" --output results/tables/m14_navier_stokes2d_medium_rollout_stability_metrics.json` | Pass | Aggregated FNO2D, SM-FNO2D v2, and SM-FNO2D v3 medium metrics. |
| `PYTHONPATH=src python3 scripts/visualize_navier_stokes2d_3d.py --prediction-path outputs/navier_stokes2d_medium_sm_fno_v3_diagnostic/predictions.npz --output-dir results/figures/m14_navier_stokes2d_v3_3d --prefix navier_stokes2d_medium_sm_fno_v3 --surface-step 17 --max-animation-frames 18 --fps 6 --dpi 120 --velocity-traces` | Pass | Wrote static surface, rollout GIF, and velocity-trace diagnostic from v3 artifact. |
| `python3 - <<'PY' ... metric/stat summary ... PY` | Pass | Confirmed one-step, rollout, per-timestep, spectrum, and stat diagnostics. |
| `python3 - <<'PY' ... PIL image inspection ... PY` | Pass | Confirmed generated M14 visualization dimensions and GIF frame count. |
| `PYTHONPATH=src pytest -q` | Pass | 44 tests passed after the final M14 changes. |
| `git diff --check` | Pass | No whitespace errors. |
| `git check-ignore -v ...` | Pass | Confirmed representative M14 data, checkpoints, metrics, predictions, and figures are ignored. |

## Metrics

Matched single-seed medium diagnostics:

| Model | Seed | One-step Relative L2 | Rollout-36 Relative L2 | Rollout Spectrum Error | Step 1 Rel L2 | Step 18 Rel L2 | Step 36 Rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FNO2D | 42 | 0.024745 | 0.081641 | 0.061890 | 0.027255 | 0.077187 | 0.122394 |
| SM-FNO2D v2 | 42 | 0.021313 | 0.763503 | 0.808335 | 0.024210 | 0.342557 | 1.666397 |
| SM-FNO2D v3 | 42 | 0.025138 | 0.079442 | 0.058582 | 0.027426 | 0.074832 | 0.119503 |

Rollout prediction and target statistics:

| Model | Prediction Mean | Prediction Std | Prediction Abs Max | Target Mean | Target Std | Target Abs Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FNO2D | 0.002149 | 0.413675 | 0.948963 | 0.000000 | 0.411428 | 0.990817 |
| SM-FNO2D v2 | 0.031164 | 0.483971 | 1.118667 | 0.000000 | 0.411428 | 0.990817 |
| SM-FNO2D v3 | 0.000861 | 0.413805 | 0.950873 | 0.000000 | 0.411428 | 0.990817 |

## Stability Assessment

SM-FNO2D v2 reproduced the one-step versus rollout mismatch: one-step
relative L2 was `0.021313`, but rollout-36 relative L2 rose to `0.763503`,
with final-step relative L2 `1.666397` and rollout spectrum error `0.808335`.
Its rollout prediction statistics also drifted from the target distribution.

SM-FNO2D v3 improved rollout stability over v2 in this diagnostic run:
rollout-36 relative L2 was `0.079442`, final-step relative L2 was `0.119503`,
and rollout spectrum error was `0.058582`. The v3 rollout prediction mean,
standard deviation, and absolute maximum stayed close to the rollout target
statistics. This is evidence that the M14 fix addresses the observed medium
v2 collapse under the current diagnostic protocol only.

## Output Artifacts

Generated local artifacts:

- `data/processed/navier_stokes2d_medium/navier_stokes2d_medium.npz`
- `results/checkpoints/navier_stokes2d_medium_fno_diagnostic.pt`
- `results/checkpoints/navier_stokes2d_medium_sm_fno_v2_diagnostic.pt`
- `results/checkpoints/navier_stokes2d_medium_sm_fno_v3_diagnostic.pt`
- `results/tables/navier_stokes2d_medium_fno_diagnostic_train_metrics.json`
- `results/tables/navier_stokes2d_medium_fno_diagnostic_eval_metrics.json`
- `results/tables/navier_stokes2d_medium_sm_fno_v2_diagnostic_train_metrics.json`
- `results/tables/navier_stokes2d_medium_sm_fno_v2_diagnostic_eval_metrics.json`
- `results/tables/navier_stokes2d_medium_sm_fno_v3_diagnostic_train_metrics.json`
- `results/tables/navier_stokes2d_medium_sm_fno_v3_diagnostic_eval_metrics.json`
- `results/tables/m14_navier_stokes2d_medium_rollout_stability_metrics.json`
- `results/tables/m14_navier_stokes2d_medium_rollout_stability_metrics.md`
- `outputs/navier_stokes2d_medium_fno_diagnostic/predictions.npz`
- `outputs/navier_stokes2d_medium_sm_fno_v2_diagnostic/predictions.npz`
- `outputs/navier_stokes2d_medium_sm_fno_v3_diagnostic/predictions.npz`
- `results/figures/navier_stokes2d_medium_fno_diagnostic_training_loss.png`
- `results/figures/navier_stokes2d_medium_fno_diagnostic_prediction.png`
- `results/figures/navier_stokes2d_medium_fno_diagnostic_rollout_relative_l2.png`
- `results/figures/navier_stokes2d_medium_sm_fno_v2_diagnostic_training_loss.png`
- `results/figures/navier_stokes2d_medium_sm_fno_v2_diagnostic_prediction.png`
- `results/figures/navier_stokes2d_medium_sm_fno_v2_diagnostic_rollout_relative_l2.png`
- `results/figures/navier_stokes2d_medium_sm_fno_v3_diagnostic_training_loss.png`
- `results/figures/navier_stokes2d_medium_sm_fno_v3_diagnostic_prediction.png`
- `results/figures/navier_stokes2d_medium_sm_fno_v3_diagnostic_rollout_relative_l2.png`
- `results/figures/m14_navier_stokes2d_v3_3d/navier_stokes2d_medium_sm_fno_v3_rollout_surface_step18.png`
- `results/figures/m14_navier_stokes2d_v3_3d/navier_stokes2d_medium_sm_fno_v3_rollout_surface_rollout.gif`
- `results/figures/m14_navier_stokes2d_v3_3d/navier_stokes2d_medium_sm_fno_v3_rollout_velocity_traces_step18.png`

Visualization dimensions:

- Static 3D-style surface PNG: `1326 x 448`.
- Rollout GIF: `1620 x 552`, 18 frames.
- Velocity trace PNG: `1608 x 497`.

These are generated artifacts under ignored local output paths.

## Limitations

- The comparison uses one seed.
- The medium dataset is a low-resolution protocol fixture.
- The v3 rollout-aware training horizon is 4 steps, while evaluation uses 36
  rollout steps.
- The 3D-style visualization renders 2D scalar vorticity as surfaces; it is not
  a true 3D flow forecast.
- Timing and stability behavior may vary with different seeds, hardware,
  larger grids, and longer training budgets.
