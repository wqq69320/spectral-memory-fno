# M12 Medium Navier-Stokes2D Diagnostic Plan

## Objective

Add a CPU-feasible medium-grid Navier-Stokes2D diagnostic that goes beyond the
16x16 smoke protocol while remaining clearly non-benchmark. The diagnostic
uses a 24x24 grid, 76 time steps, 10 input steps, one-step prediction, and a
36-step rollout horizon for FNO2D, SM-FNO2D v2, and Transformer2D where
runtime is practical.

## Scope

- Add a medium Navier-Stokes2D data config.
- Add matched medium experiment configs for FNO2D, SM-FNO2D v2, and
  Transformer2D.
- Generate the medium dataset locally.
- Run train/evaluate/plot for all feasible models, with FNO2D and SM-FNO2D v2
  required.
- Aggregate cost-style timing/parameter summaries from medium eval metrics.
- Document commands, artifacts, runtime limits, and claim boundaries in
  `docs/reports/M12_MEDIUM_NS2D_REPORT.md`.

## Done Criteria

- [x] `configs/data/navier_stokes2d_medium.yaml` exists.
- [x] Medium FNO2D, SM-FNO2D v2, and Transformer2D diagnostic configs exist.
- [x] Medium dataset generation succeeds.
- [x] FNO2D medium train/evaluate/plot succeeds.
- [x] SM-FNO2D v2 medium train/evaluate/plot succeeds.
- [x] Transformer2D medium train/evaluate/plot either succeeds or is
  explicitly deferred with runtime evidence.
- [x] Medium cost aggregate/report/plots are generated.
- [x] `PYTHONPATH=src pytest -q` passes.
- [x] Generated artifacts are confirmed git-ignored.
- [x] The final report avoids benchmark claims and model rankings.

## Non-Goals

- Do not tune final hyperparameters.
- Do not run large-grid production CFD.
- Do not make benchmark claims.
- Do not require repeated seeds unless the medium runtime remains practical.
