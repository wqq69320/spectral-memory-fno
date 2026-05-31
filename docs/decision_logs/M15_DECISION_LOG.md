# M15 Decision Log

## Decisions Made

- Added reusable dynamic-skill utilities in `src/sm_fno/evaluation/dynamic_skill.py`.
- Added `scripts/run_dynamic_skill_diagnostics.py` to analyze saved rollout
  artifacts without retraining.
- Computed model-specific persistence skill from saved prediction artifacts,
  because those artifacts contain model rollouts and matched rollout targets.
- Added held-out true temporal variation diagnostics from the medium dataset
  split to avoid relying only on one saved sample.
- Rendered the SM-FNO2D v3 step-36 3D-style surface directly from the saved v3
  rollout artifact.

## Alternatives Considered

- Re-running all model evaluations and saving full test-set rollouts would give
  broader model-specific skill diagnostics, but it would increase runtime and
  artifact size. M15 keeps the first implementation artifact-based and records
  this limitation explicitly.
- Using only aggregate M14 rollout metrics would not allow delta-prediction
  error or persistence skill from the saved rollouts, so the M15 script reads
  prediction artifacts directly.

## Implementation Rationale

Persistence skill is a direct diagnostic for the M14 concern: if targets change
very little, a low absolute rollout error can still be worse than simply
repeating the last input frame. The added diagnostics make that visible without
turning the output into a benchmark claim.

## Known Limitations

- Model-specific skill is computed from saved single-sample rollout artifacts.
- Held-out temporal variation is computed across the test trajectories, but
  full test-set model rollouts are not currently saved for every model.
- The medium Navier-Stokes2D fixture is low-resolution and synthetic.

