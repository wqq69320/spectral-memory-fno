# Figure Selection Guide

## Purpose

Identify presentation-ready generated artifacts and explain how to use them
without turning protocol-scale smoke outputs into benchmark claims. Generated
artifacts should stay ignored by git; this guide only references their paths.

## Selection Principles

- Prefer figures that explain the pipeline, not figures that imply a ranking.
- Use prediction figures to show qualitative diagnostic outputs.
- Use rollout-relative-L2 figures to explain why rollout evaluation is needed.
- Use cost plots to explain reporting mechanics and horizon-sweep structure.
- Label every metric or cost figure as a smoke/protocol sanity output.
- Do not show a leaderboard slide.

## Recommended Main-Talk Figures

| Use | Artifact Path | Why Use It | Caption Guardrail |
| --- | --- | --- | --- |
| Heat1D qualitative diagnostic | `results/figures/heat1d_sm_fno_smoke_prediction.png` | Shows that the 1D plotting path produces prediction/target diagnostics. | "Heat1D smoke diagnostic; not a benchmark result." |
| Burgers1D qualitative diagnostic | `results/figures/burgers1d_sm_fno_smoke_prediction.png` | Shows nonlinear 1D protocol coverage. | "Burgers1D smoke diagnostic; solver and model settings are protocol scale." |
| Navier-Stokes2D qualitative diagnostic | `results/figures/navier_stokes2d_sm_fno_v2_smoke_prediction.png` | Shows 2D vorticity prediction output from the v2 path. | "Navier-Stokes2D v2 smoke diagnostic; low-resolution protocol fixture." |
| Rollout error explanation | `results/figures/navier_stokes2d_sm_fno_v2_rollout20_rollout_relative_l2.png` | Shows per-step rollout behavior over a longer horizon. | "Rollout-20 sanity diagnostic; not a model ranking." |
| Cost timing by horizon | `results/figures/m11_navier_stokes2d_v2_cost/rollout_seconds_per_step_by_horizon.png` | Shows how the cost-reporting tool groups local CPU timing by horizon. | "Local CPU timing from protocol-scale runs; not final efficiency evidence." |
| Error vs timing mechanics | `results/figures/m11_navier_stokes2d_v2_cost/rollout_error_vs_seconds_per_step.png` | Shows combined error/timing reporting format. | "Use to explain reporting mechanics, not to declare a winner." |

## Backup Figures

| Topic | Artifact Path |
| --- | --- |
| Heat1D rollout-20 diagnostic | `results/figures/heat1d_sm_fno_rollout20_rollout_relative_l2.png` |
| Burgers1D rollout-20 diagnostic | `results/figures/burgers1d_sm_fno_rollout20_rollout_relative_l2.png` |
| FNO2D Navier-Stokes smoke diagnostic | `results/figures/navier_stokes2d_fno_smoke_prediction.png` |
| SM-FNO2D v1 Navier-Stokes smoke diagnostic | `results/figures/navier_stokes2d_sm_fno_smoke_prediction.png` |
| Transformer2D Navier-Stokes smoke diagnostic | `results/figures/navier_stokes2d_transformer_smoke_prediction.png` |
| M9 Heat/Burgers cost timing | `results/figures/m9_cost_efficiency/rollout_seconds_per_step_by_horizon.png` |
| M9 Heat/Burgers rollout error by horizon | `results/figures/m9_cost_efficiency/rollout_relative_l2_by_horizon.png` |

## Tables and Reports for Backup Slides

| Use | Artifact Path | Notes |
| --- | --- | --- |
| Heat/Burgers aggregate smoke metrics | `results/tables/aggregate_metrics.md` | Use only as protocol sanity evidence. |
| M9 cost report | `results/tables/m9_cost_efficiency_report.md` | Generated report for Heat1D/Burgers1D cost mechanics. |
| M10 Navier-Stokes2D aggregate | `results/tables/navier_stokes2d_aggregate_metrics.md` | Legacy Navier-Stokes2D aggregate; prefer M11 for v2. |
| M11 Navier-Stokes2D aggregate | `results/tables/navier_stokes2d_m11_aggregate_metrics.md` | Includes SM-FNO2D v2 as a separate group. |
| M11 Navier-Stokes2D v2 cost report | `results/tables/m11_navier_stokes2d_v2_cost_report.md` | Generated report for v2 cost-analysis mechanics. |

## Figures to Avoid as Main Evidence

- Any single model's training-loss curve as proof of model quality.
- Any one-seed metric table as proof of a ranking.
- Any local timing plot without the caveat that CPU load and implementation
  details affect timing.
- Any generated figure if the corresponding config and verification command
  cannot be identified.

## Suggested Figure Order

1. Heat1D prediction diagnostic.
2. Burgers1D prediction diagnostic.
3. Navier-Stokes2D prediction diagnostic.
4. Rollout relative L2 diagnostic.
5. Cost reporting by horizon.

This order tells a story about expanding protocol coverage rather than
claiming model superiority.
