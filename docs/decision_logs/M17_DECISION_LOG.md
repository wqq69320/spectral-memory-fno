# M17 Decision Log

This log captures implementation decisions and tradeoffs for M17. It is not a
benchmark interpretation.

## Decisions Made

- Evaluation now saves optional full held-out rollout artifacts in a separate
  `.npz` file controlled by `save_full_test_rollouts` and
  `full_test_rollouts_path`. The ordinary prediction artifact remains a compact
  single-sample artifact for plotting and quick inspection.
- Dynamic-skill diagnostics now prefer the full-test rollout artifact when it is
  available and fall back to the saved-sample artifact for older M15/M16 runs.
- Training now supports weighted delta and persistence-aware loss terms through
  `delta_loss_weight` and `persistence_loss_weight`, alongside the existing
  `rollout_train_steps` and `rollout_loss_weight` controls.
- Added `configs/experiment/navier_stokes2d_dynamic_sm_fno_v3_skill_optimized.yaml`
  with `epochs: 4`, `rollout_train_steps: 8`, `rollout_loss_weight: 0.3`,
  `delta_loss_weight: 0.2`, `persistence_loss_weight: 0.1`, and a CPU-sized
  SM-FNO2D v3 model.
- Implemented residual FNO base initialization for SM-FNO2D v3 through
  `fno_base_checkpoint_path`. The M17 optimized config initializes from
  `results/checkpoints/navier_stokes2d_dynamic_fno_diagnostic.pt`.

## Alternatives Considered

- Full-test-set rollout saving in the main evaluation artifact vs a separate
  artifact. A separate artifact was used to avoid making every evaluation output
  large.
- Delta MSE loss vs persistence-relative loss vs a weighted combination with
  ordinary rollout MSE. M17 uses a weighted combination so the objective keeps a
  standard forecasting term while making trajectory change and persistence
  parity explicit.
- Initializing only the FNO residual base from a trained checkpoint vs training
  the full SM-FNO2D v3 model from scratch. M17 initializes only the residual FNO
  base because SM-FNO2D v3 stores it as `model.base` with the same FNO2D state
  dict structure.

## Implementation Rationale

M16 showed that single-sample dynamic skill is too weak for interpreting
persistence skill. M17 uses full held-out rollouts so mean skill is computed
across held-out trajectories and rollout steps.

Persistence-aware or delta-prediction losses should make the training objective
closer to the question being tested: whether the model learns meaningful change
beyond repeating the last input frame.

The FNO base checkpoint initialization is intentionally explicit and
config-driven. If the configured checkpoint is missing, training raises an error
instead of silently falling back to random initialization.

## Known Limitations

- The dynamic fixture is synthetic and low-resolution.
- Longer rollout-aware training is still CPU-limited; the M17 config uses four
  epochs rather than a large tuning budget.
- Full-test-set rollout artifacts may be larger and must remain ignored.
- Positive skill on this fixture would still be a diagnostic result, not a
  benchmark claim.
- `strict=False` is used when loading the residual FNO base so future compatible
  FNO state dict extensions can be inspected through saved missing/unexpected
  key metadata.

## Follow-Up Items

- Add full-test-set skill metrics to aggregate reports if M17 proves the
  interface useful.
- Consider repeated seeds after the single-seed M17 diagnostic is stable.
- Consider stronger dynamic fixtures with forcing or larger grids in a later
  milestone.
