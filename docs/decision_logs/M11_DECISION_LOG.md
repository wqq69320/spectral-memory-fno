# M11 Decision Log

## Decisions Made

- Clamp `SpectralConv2D` retained top and bottom y-frequency counts so the
  positive and negative frequency writes never overlap on small or odd grids.
- Keep FNO2D invalid-grid handling permissive by clamping modes rather than
  rejecting CPU-friendly grids such as 8x8 or 15x15.
- Add `StableGatedDiagonalSSM` with transition values constrained to `(0, 1)`
  through `exp(-softplus(log_decay))`.
- Use a residual gated temporal-memory output so SM-FNO2D v2 can blend memory
  output with the current hidden state.
- Add `SpectralMemoryFNO2DV2` as a separate model class and config option
  instead of modifying the original SM-FNO2D implementation in place.
- Keep Navier-Stokes2D v2 smoke and rollout-20 configs separate from the M10
  v1 configs.
- Extend the M9/M10 cost-efficiency horizon sweep naming to support an M11
  analysis prefix and report `sm_fno2d_v2` separately.

## Alternatives Considered

- Rejecting `height < 2 * modes` was rejected because clamping preserves valid
  small-grid smoke configs while making retained spectral rows explicit.
- Replacing SM-FNO2D v1 with the v2 memory path was rejected because M11 needs
  to preserve the M10 baseline behavior for side-by-side protocol checks.
- A larger Navier-Stokes2D grid and longer training budget were deferred to
  keep M11 verification CPU-friendly.
- Adding more seeds was deferred because the current two-seed sweep validates
  reporting mechanics only.

## Rationale

- The FNO2D fix directly addresses the review finding: bottom y-mode writes no
  longer overwrite rows already written by top y modes.
- Stable transition parameterization prevents the temporal memory block from
  learning unconstrained recurrent multipliers during smoke-scale training.
- A separate v2 class makes the comparison auditable: v1, v2, FNO2D, and
  Transformer2D all remain config-selectable.
- Reporting v2 as a distinct cost group prevents aggregation from mixing old
  and new model variants.

## Known Limitations

- SM-FNO2D v2 is a limited stabilization experiment, not a finalized model.
- The Navier-Stokes2D generator remains a lightweight low-resolution protocol
  fixture, not a validated high-accuracy CFD solver.
- Timing values are local CPU wall-clock measurements.
- Repeated seeds `0` and `1` validate reproducibility plumbing only.
- M11 smoke, rollout, repeated-seed, aggregate, and cost-efficiency outputs are
  sanity checks only and must not be interpreted as benchmark claims or model
  rankings.

## Follow-Up Work

- Add larger 2D grids and longer training budgets once protocol-scale behavior
  is stable.
- Run more repeated seeds before drawing any comparative conclusions.
- Add stronger numerical diagnostics for Navier-Stokes trajectories.
- Consider additional memory parameterizations only after v2 behavior is
  characterized under a larger protocol.
