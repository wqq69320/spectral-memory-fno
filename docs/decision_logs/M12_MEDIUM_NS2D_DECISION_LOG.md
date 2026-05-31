# M12 Medium Navier-Stokes2D Diagnostic Decision Log

## Decisions Made

- Use a 24x24 grid instead of 32x32 for the first medium diagnostic to keep CPU
  training and rollout feasible in the local workflow.
- Use 76 time steps, 10 input steps, one predicted step, and 36 rollout steps.
- Use 36 trajectories so train, validation, and test splits remain available
  without making the dataset generation or training loop unnecessarily large.
- Keep FNO2D, SM-FNO2D v2, and Transformer2D on matched data, seed, split,
  training budget, and rollout horizon.
- Skip repeated seeds unless the single-seed medium path proves fast enough.
- Reuse existing train, evaluate, plot, aggregate-cost, cost-plot, and
  generated-report scripts.

## Rationale

- The diagnostic should test longer-horizon 2D protocol stability, not final
  model quality.
- 24x24 gives 2.25x as many grid points as the 16x16 smoke fixture while
  staying CPU-feasible.
- A 36-step rollout is long enough to exercise autoregressive behavior beyond
  the M10/M11 rollout-20 protocol.
- Reusing existing scripts provides stronger evidence that the protocol scales
  than adding a separate one-off runner.

## Known Limitations

- The medium dataset is still a low-resolution protocol fixture.
- The training budget is intentionally small.
- Single-seed results cannot support statistical conclusions.
- Local CPU timing is not hardware-independent.
- Cost summaries are diagnostic artifacts only.

## Follow-Up Work

- Increase seeds after runtime is characterized.
- Try 32x32 or larger grids once the 24x24 path is stable.
- Add physical diagnostics such as energy and spectrum errors.
- Add memory and FLOP measurements before any cost-efficiency claim.
