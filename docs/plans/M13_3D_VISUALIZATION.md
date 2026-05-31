# M13 3D-Style Navier-Stokes2D Visualization Plan

## Objective

Create presentation-ready visual diagnostics from existing Navier-Stokes2D
prediction artifacts. The outputs should render 2D vorticity as 3D-style
surfaces, true/predicted/error panels, and a rollout animation while avoiding
any claim that the underlying data or model is truly 3D.

## Scope

- Add a reusable visualization helper module.
- Add a CLI script for rendering from `predictions.npz` artifacts.
- Generate at least one static 3D-style surface figure.
- Generate at least one rollout animation.
- Add an optional vorticity-derived velocity trace diagnostic.
- Write `docs/reports/M13_3D_VISUALIZATION_REPORT.md`.
- Verify tests, artifact-ignore behavior, and no-claim guardrails.

## Done Criteria

- [x] Static true/predicted/error 3D-style vorticity surface figure generated.
- [x] Rollout GIF animation generated.
- [x] Optional velocity-trace diagnostic generated and documented.
- [x] Generated artifacts are under ignored output paths.
- [x] `PYTHONPATH=src pytest -q` passes.
- [x] Report documents commands, artifact paths, limitations, and claim
  boundaries.

## Non-Goals

- Do not add true 3D Navier-Stokes data.
- Do not claim 3D flow forecasting.
- Do not make benchmark claims from visualization artifacts.
