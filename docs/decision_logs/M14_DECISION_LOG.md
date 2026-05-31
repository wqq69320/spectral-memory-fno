# M14 Decision Log

## Decisions Made

- Add `SpectralMemoryFNO2DV3` instead of modifying SM-FNO2D v2 in place.
- Make v3 an FNO2D residual base predictor plus a conservative gated temporal
  correction.
- Initialize the v3 correction gate with a negative bias and cap it with
  `gate_limit`, so the SSM path starts as a small correction rather than a
  replacement forecast.
- Reuse `StableGatedDiagonalSSM` for the temporal correction path.
- Add optional rollout-aware training at the trainer level with
  `rollout_train_steps` and `rollout_loss_weight`.
- Keep rollout-aware training disabled unless explicitly configured.
- Extend evaluation diagnostics with prediction/target statistics and spatial
  Fourier spectrum error for one-step and rollout outputs.
- Use the existing medium 24x24 Navier-Stokes2D protocol to compare FNO2D,
  SM-FNO2D v2, and SM-FNO2D v3.

## Alternatives Considered

- Replacing v2 with v3 was rejected because M14 needs a direct diagnostic
  comparison and must preserve M11/M12 behavior.
- Training v3 with only one-step loss was rejected because the failure mode is
  specifically autoregressive rollout drift.
- A long rollout-aware training horizon was deferred because it would make the
  CPU diagnostic substantially slower.
- Adding a new data fixture was rejected because the observed collapse already
  occurs under the existing medium protocol.

## Rationale

- FNO2D is stable under the medium rollout diagnostic, so using it as the base
  forecast gives v3 a conservative default trajectory.
- The SSM branch can still learn temporal corrections, but the learned gate and
  gate cap reduce the risk that one-step improvements destabilize rollout.
- Rollout-aware training makes the training objective expose short-horizon
  autoregressive feedback while keeping the full 36-step evaluation untouched.
- Keeping v3 as a new model name makes aggregation and reports auditable.

## Known Limitations

- The medium fixture remains low resolution and CPU-focused.
- M14 uses a single seed for the matched FNO2D/v2/v3 comparison.
- The rollout-aware training horizon is 4 steps, shorter than the 36-step
  evaluation horizon.
- Timing values are local CPU wall-clock measurements.
- M14 diagnostics show whether v3 fixes the observed v2 collapse in this
  protocol run only; they are not benchmark or model-ranking evidence.

## Follow-Up Work

- Run repeated seeds for v3 once the single-seed diagnostic is accepted.
- Test longer rollout-aware training horizons and larger grids.
- Add physical diagnostics such as energy spectra under a stricter numerical
  protocol.
- Consider exporting v3 correction-gate statistics for deeper ablation work.
