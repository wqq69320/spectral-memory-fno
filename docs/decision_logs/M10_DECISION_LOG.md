# M10 Decision Log

## Decisions Made

- Use a pseudo-spectral periodic vorticity generator with explicit nonlinear
  advection and semi-implicit diffusion for Navier-Stokes2D.
- Keep the default Navier-Stokes2D dataset CPU-friendly: 64 trajectories, a
  16x16 grid, and 36 time steps.
- Enforce an explicit advection CFL guard in the generator and raise before
  writing non-finite trajectories.
- Preserve the 2D tensor shape convention
  `[batch, time, height, width, channels] -> [batch, pred_time, height, width, channels]`.
- Add compact FNO2D, SM-FNO2D, and Transformer2D implementations rather than
  overloading the 1D classes.
- Reuse existing train/evaluate/plot/repeated-seed/aggregation scripts wherever
  possible.
- Add a Navier-Stokes-specific M10 horizon-sweep analysis config for the M9
  cost-efficiency runner.
- Keep smoke and rollout-20 configs separate so longer rollout validation does
  not alter the smoke protocol.

## Alternatives Considered

- A high-resolution Navier-Stokes solver was deferred because M10 is a protocol
  extension milestone, not a CFD validation milestone.
- A velocity-pressure formulation was deferred in favor of vorticity form,
  which avoids pressure solves and gives a compact scalar field target.
- Spatial attention over all 2D grid cells was deferred for the Transformer2D
  baseline; M10 uses temporal attention per grid cell to match the existing 1D
  baseline design and keep CPU cost small.
- Static horizon-sweep YAML files for every seed and horizon were deferred
  because the existing horizon-sweep runner already expands configs clearly.

## Rationale

- The pseudo-spectral vorticity generator is simple to inspect and produces
  smooth periodic fields suitable for protocol validation.
- Separate 2D model classes keep shape handling explicit and reduce risk of
  silently treating height as time or channels.
- Reusing existing scripts verifies that the established protocol generalizes
  to 2D without creating a parallel workflow.
- Small model widths and grid sizes make the verification commands practical on
  CPU.

## Known Limitations

- The Navier-Stokes generator is a lightweight protocol fixture, not a validated
  high-accuracy solver.
- The generated fields are smooth and low resolution.
- Timing values are local CPU wall-clock measurements.
- Repeated seeds `0` and `1` validate reporting mechanics only.
- Smoke, rollout, repeated-seed, aggregate, and cost-efficiency outputs are
  sanity checks only and must not be interpreted as benchmark claims or model
  rankings.

## Follow-Up Work

- Add stronger numerical diagnostics for vorticity trajectories.
- Add larger grids, longer horizons, and more seeds only after protocol-scale
  stability is established.
- Consider velocity-channel forecasting and pressure/energy diagnostics in a
  later milestone.
- Add final benchmark reports only when larger configs, metrics, plots, and
  verification commands support the claims being made.
