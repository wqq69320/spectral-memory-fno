# M16 Dynamic Navier-Stokes2D Fixture Plan

## Objective

Create a more dynamic but still CPU-feasible Navier-Stokes2D diagnostic fixture
so the 36-step rollout evaluation is not dominated by near-stationary targets.
Run FNO2D, SM-FNO2D v3, and persistence diagnostics on the new fixture.

## Scope

- Add config-driven Navier-Stokes2D `save_every`, `initial_amplitude`, and
  `max_modes` controls while preserving the internal CFL stability check.
- Add a dynamic 24x24 Navier-Stokes2D fixture config.
- Add dynamic FNO2D and SM-FNO2D v3 diagnostic experiment configs.
- Generate dynamic data and verify finite stable trajectories.
- Train/evaluate/plot FNO2D and SM-FNO2D v3 under a 36-step rollout.
- Run persistence and dynamic-skill diagnostics on the dynamic artifacts.
- Render a step-36 3D-style surface for SM-FNO2D v3.
- Document commands, metrics, artifacts, limitations, and interpretation.

## Non-Goals

- Do not claim benchmark wins.
- Do not tune hyperparameters beyond a CPU-feasible diagnostic setup.
- Do not treat the synthetic low-resolution fixture as validated CFD evidence.

## Done Criteria

- Dynamic dataset generation succeeds without CFL or finite-value failures.
- Held-out persistence error is meaningfully above near-zero over the 36-step
  horizon.
- FNO2D and SM-FNO2D v3 train/evaluate/plot successfully.
- Dynamic-skill diagnostics produce JSON, Markdown, plots, and a v3 step-36
  surface artifact.
- Tests pass.
- Generated artifacts remain git-ignored.
- `docs/reports/M16_DYNAMIC_NS2D_REPORT.md` records interpretation boundaries.

