# M8 Decision Log

## Decisions Made

- Replace the Burgers1D random placeholder with a deterministic periodic
  viscous Burgers generator.
- Use smooth low-frequency random initial conditions, matching the Heat1D
  generator's reproducibility style.
- Use a conservative Rusanov flux for nonlinear advection and centered finite
  differences for diffusion.
- Enforce the combined explicit convection-diffusion condition
  `CFL + 2 * diffusion <= stability_safety` before each Burgers update, using
  a default safety margin of `0.95`.
- Keep the default Burgers1D data config smoke-sized: 128 samples, 64 grid
  points, and 50 time steps.
- Reuse the existing train/evaluate/plot/repeated-seed/aggregation scripts for
  Burgers1D instead of creating a separate workflow.
- Add smoke and rollout-20 configs for FNO, SM-FNO, and Transformer.
- Keep generated Burgers outputs in the same ignored `data/`, `outputs/`, and
  `results/` locations.

## Alternatives Considered

- A spectral Burgers solver was deferred to keep the implementation simple and
  easy to inspect.
- A higher-resolution or longer-time Burgers dataset was deferred because M8
  verification must remain CPU-friendly.
- Adding an MLP Burgers config was deferred because the M8 scope requested FNO,
  SM-FNO, and Transformer configs.
- Implementing shock-capturing high-resolution schemes was deferred; the M8
  goal is protocol extension rather than production CFD accuracy.

## Rationale

- The Rusanov-plus-diffusion scheme is robust enough for smooth smoke data and
  simple to validate.
- Matching Heat1D protocol fields keeps cross-PDE comparisons reproducible
  without implying benchmark quality.
- Running the existing model configs on Burgers validates that the M6/M7
  protocol can extend beyond Heat1D.

## Known Limitations

- The Burgers solver is a CPU-friendly explicit finite-difference generator
  with a combined stability guard, not a validated high-accuracy solver.
- The generated dataset is small and smooth.
- Two repeated seeds are enough to verify protocol mechanics only.
- Rollout-20 configs are protocol-validation configs, not final benchmark
  configs.
- Metric values are sanity outputs and must not be interpreted as model
  rankings.

## Follow-Up Work

- Add stronger numerical validation for Burgers trajectories.
- Add larger seed sets and longer horizons only after smoke protocol stability.
- Add higher-resolution Burgers configs for future non-smoke experiments.
- Consider additional diagnostics such as conservation and shock-sensitive
  errors in later milestones.
