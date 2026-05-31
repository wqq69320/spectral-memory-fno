# M17 Dynamic Skill Optimization Plan

## Objective

Improve SM-FNO2D v3 on the dynamic Navier-Stokes2D fixture using full-test-set
dynamic skill diagnostics and persistence-aware training. The goal is not to
make benchmark claims, but to test whether SM-FNO can learn meaningful dynamics
beyond last-frame persistence.

M16 established that the dynamic fixture is more persistence-hard than the
medium fixture, but model-specific skill was still computed from saved
single-sample rollout artifacts. M17 closes that gap by saving and evaluating
full held-out rollouts.

## Scope

- Add optional full-test-set rollout saving in `scripts/evaluate.py`.
- Compute full-test-set dynamic skill aggregation:
  - model rollout relative L2,
  - persistence rollout relative L2,
  - delta-prediction relative L2,
  - persistence-normalized skill.
- Add a persistence-aware or delta-prediction loss option to the training loop.
- Add a stronger dynamic SM-FNO2D config using:
  - longer training than M16,
  - `rollout_train_steps` in the `8-12` range,
  - `rollout_loss_weight` around `0.3`,
  - CPU-feasible model size.
- Run FNO2D and improved SM-FNO2D on the dynamic fixture under the same
  36-step rollout protocol.
- If feasible, initialize the FNO residual base in SM-FNO2D from a trained FNO2D
  checkpoint. If not feasible within the current code structure, document why
  it is deferred and what interface would be needed.
- Generate updated figures:
  - rollout error vs persistence,
  - persistence-normalized skill,
  - delta-prediction error,
  - step-36 3D-style surface for the improved SM-FNO2D artifact.
- Report whether the improved SM-FNO2D achieves positive mean skill over
  persistence on held-out trajectories.

## Non-Goals

- Do not claim benchmark wins.
- Do not compare as a final architecture ranking.
- Do not tune large grids or expensive training budgets.
- Do not treat the synthetic dynamic fixture as validated CFD evidence.
- Do not remove M15/M16 diagnostics; M17 should extend them.

## Architecture Requirements

- Keep the dynamic skill implementation config-driven and CPU-feasible.
- Preserve existing M16 configs and artifacts.
- Store full-test-set rollout artifacts only under ignored output paths.
- Keep dynamic skill metrics clearly separated from ordinary one-step metrics.
- Use plain JSON/Markdown outputs for reproducibility.

## Proposed Implementation Steps

1. Extend evaluation with an optional config field such as
   `save_full_test_rollouts: true`.
2. Save full held-out rollout inputs, targets, predictions, and persistence
   predictions to an ignored `.npz` artifact.
3. Extend or add a diagnostic script that reads full-test rollout artifacts and
   writes aggregate skill metrics over all held-out trajectories.
4. Add training loss options such as:
   - `loss_mode: mse`,
   - `loss_mode: delta_mse`,
   - `persistence_loss_weight`,
   - `delta_loss_weight`.
5. Add an improved SM-FNO2D v3 dynamic config, for example:
   - `epochs: 6-10`,
   - `rollout_train_steps: 8-12`,
   - `rollout_loss_weight: 0.3`,
   - optional persistence/delta loss weights.
6. Add tests for full-rollout artifact shape and dynamic skill aggregation.
7. Run FNO2D and improved SM-FNO2D v3 diagnostics on the dynamic fixture.
8. Update the M17 decision log and verification report.

## Verification Commands

These commands should be run or updated with final config names during
implementation:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/navier_stokes2d_dynamic.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.yaml
PYTHONPATH=src python3 scripts/run_dynamic_skill_diagnostics.py --experiment-config configs/experiment/navier_stokes2d_dynamic_fno_diagnostic.yaml --experiment-config configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.yaml --output-json results/tables/m17_dynamic_skill_optimization.json --output-md results/tables/m17_dynamic_skill_optimization.md --figures-dir results/figures/m17_dynamic_skill_optimization --artifact-prefix m17_dynamic_skill --title-prefix M17 --v3-experiment navier_stokes2d_dynamic_sm_fno_v3_skill_optimized
git diff --check
```

## Done Criteria

- `pytest` passes.
- Full-test-set rollout skill diagnostics run successfully.
- Improved SM-FNO2D train/evaluate/plot works.
- Persistence skill is reported over held-out trajectories, not only one saved
  sample.
- Generated artifacts remain ignored.
- `docs/reports/M17_VERIFICATION_REPORT.md` documents metrics, commands,
  artifacts, limitations, and whether dynamic skill improved.

