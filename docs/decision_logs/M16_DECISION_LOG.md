# M16 Decision Log

## Decisions Made

- Added `save_every`, `initial_amplitude`, and `max_modes` parameters to the
  Navier-Stokes2D generator.
- Kept `dt` as the internal solver step and used `save_every` to increase the
  saved-frame spacing while preserving the existing explicit advection CFL
  check.
- Chose a 24x24 dynamic fixture with `dt=0.004`, `save_every=10`,
  `viscosity=0.0005`, `initial_amplitude=1.25`, and `max_modes=5`.
- Reused the M15 dynamic-skill diagnostic script with configurable experiment
  paths and M16 artifact naming.
- Evaluated only FNO2D and SM-FNO2D v3 for M16, plus persistence diagnostics,
  because the goal is to test persistence hardness rather than add another
  baseline.

## Alternatives Considered

- Increasing the grid to 32x32 would likely provide richer dynamics, but the
  24x24 fixture keeps the run CPU-feasible and comparable to the M14/M15 medium
  protocol.
- Adding explicit forcing would create another control knob, but increasing
  saved-frame spacing and initial condition amplitude was enough to raise
  horizon persistence error above near-zero.

## Implementation Rationale

The previous medium fixture had very small step-to-step and horizon changes.
Using `save_every` decouples numerical stability from saved-frame dynamics:
the solver still advances with a small stable internal `dt`, while the dataset
stores states separated by a larger effective interval.

## Known Limitations

- The dynamic fixture is still synthetic, low-resolution, and single-seed.
- Per-saved-step true change remains modest, even though 36-step accumulated
  persistence error is no longer near-zero.
- Dynamic-skill metrics are computed from saved single-sample rollout artifacts;
  held-out persistence diagnostics cover the test trajectories.

