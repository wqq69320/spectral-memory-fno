# M17 Verification Report

M17 outputs are diagnostic artifacts only. They are not benchmark claims, model
rankings, or evidence of general PDE forecasting performance.

## Objective

Improve SM-FNO2D v3 on the dynamic Navier-Stokes2D fixture using full-test-set
dynamic skill diagnostics and persistence-aware training. The report must state
whether the improved model achieves positive mean skill over last-frame
persistence on held-out trajectories.

## Implementation Summary

- `scripts/evaluate.py` now supports optional full-test-set rollout saving via
  `save_full_test_rollouts` and `full_test_rollouts_path`.
- Full-test rollout artifacts store held-out inputs, rollout targets, rollout
  predictions, and persistence predictions under ignored `outputs/` paths.
- `scripts/run_dynamic_skill_diagnostics.py` now prefers full-test rollout
  artifacts and computes model error, persistence error, delta-prediction error,
  and persistence-normalized skill over held-out trajectories.
- `Trainer` now supports `delta_loss_weight` and `persistence_loss_weight` in
  addition to rollout-aware training loss.
- Added an optimized diagnostic config for SM-FNO2D v3 with 4 epochs,
  `rollout_train_steps: 8`, `rollout_loss_weight: 0.3`,
  `delta_loss_weight: 0.2`, and `persistence_loss_weight: 0.1`.
- FNO residual-base initialization was implemented through
  `fno_base_checkpoint_path`; the optimized SM-FNO2D run initialized from the
  trained FNO2D diagnostic checkpoint.

## Commands Run

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q` | Pass | 53 tests passed. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/navier_stokes2d_dynamic.yaml` | Pass | Generated shape `(36, 76, 24, 24, 1)`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml` | Pass | Final train loss `0.000664`. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml` | Pass | Saved full-test rollouts; rollout relative L2 `0.145452`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.yaml` | Pass | Final train loss `0.000023`; FNO base initialized. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.yaml` | Pass | Saved full-test rollouts; rollout relative L2 `0.025526`. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.yaml` | Pass | Saved training, prediction, and rollout plots. |
| `PYTHONPATH=src python3 scripts/run_dynamic_skill_diagnostics.py --experiment-config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml --experiment-config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.yaml --output-json results/tables/m17_dynamic_skill_optimization.json --output-md results/tables/m17_dynamic_skill_optimization.md --figures-dir results/figures/m17_dynamic_skill_optimization --artifact-prefix m17_dynamic_skill --title-prefix M17 --v3-experiment navier_stokes2d_dynamic_sm_fno_v3_skill_optimized` | Pass | Full-test dynamic skill diagnostics and figures generated. |
| `git diff --check` | Pass | No whitespace errors. |
| `git check-ignore -v ... representative M17 artifacts ...` | Pass | Data, checkpoints, outputs, tables, and figures remain ignored. |

## Metrics

Fill this section from full-test-set held-out rollout diagnostics.

| Model | Full-Test Model Rel L2 Mean | Full-Test Persistence Rel L2 Mean | Full-Test Delta Error Mean | Full-Test Skill Mean | Full-Test Skill Step 36 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FNO2D | 0.139531 | 0.127381 | 1.528133 | -0.461319 | 0.051239 |
| SM-FNO2D v3 optimized | 0.024568 | 0.127381 | 0.263177 | 0.743001 | 0.837336 |

## Dynamic Skill Interpretation

The optimized SM-FNO2D v3 diagnostic achieved positive mean skill over
last-frame persistence on the full held-out dynamic fixture rollouts:
`0.743001` mean skill and `0.837336` step-36 skill.

This is an M17 diagnostic result only. It shows that, for this single
CPU-feasible dynamic fixture and seed, the persistence-aware SM-FNO2D v3 run
learned nontrivial trajectory change beyond last-frame repetition. It is not a
benchmark claim, not a general ranking against attention baselines, and not
evidence of broad Navier-Stokes forecasting performance.

## Output Artifacts

Data and checkpoints:

- `data/processed/navier_stokes2d_dynamic/navier_stokes2d_dynamic.npz`
- `results/checkpoints/navier_stokes2d_dynamic_fno_diagnostic.pt`
- `results/checkpoints/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.pt`

Metrics:

- `results/tables/navier_stokes2d_dynamic_fno_diagnostic_train_metrics.json`
- `results/tables/navier_stokes2d_dynamic_fno_diagnostic_eval_metrics.json`
- `results/tables/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized_train_metrics.json`
- `results/tables/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized_eval_metrics.json`
- `results/tables/m17_dynamic_skill_optimization.json`
- `results/tables/m17_dynamic_skill_optimization.md`

Figures:

- `results/figures/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized_training_loss.png`
- `results/figures/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized_prediction.png`
- `results/figures/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized_rollout_relative_l2.png`
- `results/figures/m17_dynamic_skill_optimization/m17_dynamic_skill_rollout_vs_persistence.png`
- `results/figures/m17_dynamic_skill_optimization/m17_dynamic_skill_persistence_skill.png`
- `results/figures/m17_dynamic_skill_optimization/m17_dynamic_skill_delta_prediction_error.png`
- `results/figures/m17_dynamic_skill_optimization/m17_dynamic_skill_heldout_temporal_variation.png`
- `results/figures/m17_dynamic_skill_optimization/m17_dynamic_skill_sm_fno2d_v3_surface_step36.png`

## Limitations

- Single dynamic fixture unless repeated seeds are added.
- Synthetic low-resolution Navier-Stokes2D diagnostic only.
- CPU-feasible training budget may limit conclusions.
- Positive or negative skill is a protocol diagnostic, not a general benchmark
  result.
- The FNO2D and optimized SM-FNO2D configurations have different objectives, so
  the table is a protocol sanity check rather than a controlled architecture
  ranking.

## Remaining Risks

- Full-test rollout artifacts are larger than single-sample prediction files,
  so future larger grids may require a streaming metric path.
- The optimized run uses one seed; repeated-seed aggregation is still needed
  before making stronger research statements.
- Positive skill may depend on the synthetic dynamic fixture settings and the
  FNO-base checkpoint initialization.
