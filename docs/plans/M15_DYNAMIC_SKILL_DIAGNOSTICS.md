# M15 Persistence Baseline and Dynamic Skill Diagnostics

## Objective

Verify whether the low medium Navier-Stokes2D rollout errors from M14 are
meaningful or partly explained by nearly stationary targets. Add a persistence
baseline, temporal variation diagnostics, delta-prediction error, and
persistence-normalized skill score.

## Scope

- Analyze saved medium Navier-Stokes2D rollout artifacts for:
  - FNO2D
  - SM-FNO2D v2
  - SM-FNO2D v3
  - Transformer2D, where available
- Repeat the last input frame as the persistence baseline over the 36-step
  rollout horizon.
- Compute per-step model error, persistence error, true temporal change,
  delta-prediction error, and persistence-normalized skill.
- Compute held-out true trajectory temporal diagnostics from the medium dataset
  split.
- Generate comparison plots and a step-36 3D-style surface for v3.
- Update `docs/reports/M15_DYNAMIC_SKILL_REPORT.md`.

## Non-Goals

- Do not retrain models unless the existing artifacts are missing or invalid.
- Do not make benchmark claims or model rankings.
- Do not treat the synthetic medium fixture as validated CFD evidence.

## Done Criteria

- M15 diagnostics run successfully on available medium artifacts.
- Tests pass.
- M15 JSON, Markdown, and plots are generated under ignored artifact paths.
- Step-36 v3 3D-style surface is rendered.
- Report records commands, metrics, artifact paths, limitations, and whether v3
  is meaningfully better than persistence.

