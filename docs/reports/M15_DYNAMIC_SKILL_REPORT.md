# M15 Dynamic Skill Report

M15 outputs are diagnostic artifacts only. They are not benchmark claims, model
rankings, or evidence of general PDE forecasting performance.

## Objective

M14 fixed the visible SM-FNO2D v2 rollout collapse in the medium
Navier-Stokes2D protocol. M15 checks whether low rollout error is meaningful or
partly explained by nearly stationary rollout targets by adding:

- persistence baseline error from repeating the last input frame,
- true temporal variation diagnostics,
- delta-prediction error,
- persistence-normalized skill score,
- comparison plots over the 36-step rollout horizon,
- a step-36 3D-style surface rendering for SM-FNO2D v3.

## Commands Run

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q tests/test_dynamic_skill.py tests/test_metrics.py tests/test_shapes.py` | Pass | 28 focused tests passed. |
| `python3 -m py_compile scripts/run_dynamic_skill_diagnostics.py src/sm_fno/evaluation/dynamic_skill.py` | Pass | New script and utility module compile. |
| `PYTHONPATH=src python3 scripts/run_dynamic_skill_diagnostics.py` | Pass | Wrote M15 JSON, Markdown, plots, and v3 step-36 surface. |
| `python3 - <<'PY' ... inspect m15_dynamic_skill_diagnostics.json ... PY` | Pass | Confirmed metrics and plot paths were written. |
| `python3 - <<'PY' ... PIL image inspection ... PY` | Pass | Confirmed generated M15 PNG dimensions. |
| `PYTHONPATH=src pytest -q` | Pass | 49 tests passed after M15 changes. |
| `git diff --check` | Pass | No whitespace errors. |
| `git check-ignore -v ... representative M15 artifacts ...` | Pass | Confirmed generated M15 tables/figures and existing data/checkpoint/prediction artifacts are ignored. |

## Diagnostic Scope

Model-specific dynamic-skill metrics are computed from saved medium prediction
artifacts:

- `outputs/navier_stokes2d_medium_fno_diagnostic/predictions.npz`
- `outputs/navier_stokes2d_medium_sm_fno_v2_diagnostic/predictions.npz`
- `outputs/navier_stokes2d_medium_sm_fno_v3_diagnostic/predictions.npz`
- `outputs/navier_stokes2d_medium_transformer_diagnostic/predictions.npz`

Held-out true temporal variation is computed from the medium dataset split used
by the diagnostic configs:

- `data/processed/navier_stokes2d_medium/navier_stokes2d_medium.npz`

## Metrics

Persistence-normalized skill is:

```text
skill = 1 - model_error / persistence_error
```

Positive values mean lower error than persistence. Zero is parity with
persistence. Negative values mean worse than persistence.

The following values are from saved single-sample rollout artifacts:

| Model | Model Rel L2 Mean | Model Rel L2 Step 36 | Persistence Rel L2 Mean | Persistence Rel L2 Step 36 | Skill Mean | Skill Step 36 | Delta Error Mean | Delta Error Step 36 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FNO2D | 0.089019 | 0.142437 | 0.019480 | 0.037844 | -5.222869 | -2.763740 | 6.222882 | 3.763741 |
| SM-FNO2D v2 | 0.644869 | 2.014222 | 0.019480 | 0.037844 | -25.529959 | -52.223724 | 26.529978 | 53.223743 |
| SM-FNO2D v3 | 0.081244 | 0.128584 | 0.019480 | 0.037844 | -4.795699 | -2.397685 | 5.795711 | 3.397686 |
| Transformer2D | 0.224650 | 0.382304 | 0.019480 | 0.037844 | -13.396210 | -9.101994 | 14.396236 | 10.101996 |

Held-out true temporal variation over 5 medium test trajectories:

| Diagnostic | Mean | Step 36 |
| --- | ---: | ---: |
| Persistence error / change from last input | 0.019464 | 0.037708 |
| True step-to-step change | 0.001034 | 0.001008 |

## Interpretation

Under the saved sample-artifact diagnostic, SM-FNO2D v3 is **not meaningfully
better than persistence**. Its mean skill is `-4.795699` and its step-36 skill
is `-2.397685`, both below the predeclared positive-skill threshold. This means
the low absolute rollout error from M14 must be interpreted alongside the small
true temporal changes in the medium fixture.

This does not invalidate the M14 observation that v3 fixed the visible v2
collapse. It narrows the interpretation: v3 is much more stable than v2 in the
saved rollout artifact, but M15 does not show that v3 beats a last-frame
persistence baseline on this medium diagnostic sample.

## Output Artifacts

Tables:

- `results/tables/m15_dynamic_skill_diagnostics.json`
- `results/tables/m15_dynamic_skill_diagnostics.md`

Figures:

- `results/figures/m15_dynamic_skill/m15_rollout_vs_persistence.png`
- `results/figures/m15_dynamic_skill/m15_heldout_temporal_variation.png`
- `results/figures/m15_dynamic_skill/m15_persistence_skill.png`
- `results/figures/m15_dynamic_skill/m15_delta_prediction_error.png`
- `results/figures/m15_dynamic_skill/m15_sm_fno2d_v3_surface_step36.png`

Generated figure dimensions:

- Rollout vs persistence: `1440 x 864`
- Held-out temporal variation: `1440 x 864`
- Persistence skill: `1440 x 864`
- Delta-prediction error: `1440 x 864`
- SM-FNO2D v3 step-36 surface: `1545 x 522`

These are generated artifacts under ignored local output paths.

## Limitations

- Model-specific persistence skill uses saved single-sample prediction artifacts.
- Held-out temporal variation is broader than the saved sample, but full
  test-set model rollouts are not currently stored for every model.
- The medium Navier-Stokes2D fixture has small true step-to-step change, which
  makes persistence a strong baseline.
- This is a low-resolution synthetic diagnostic protocol, not validated CFD
  evidence.
- No benchmark claims should be made from M15.

## Remaining Risks

- Future diagnostics should optionally save full test-set rollout predictions
  to compute persistence skill across all held-out trajectories.
- Dynamic-skill conclusions may change with higher-viscosity/lower-viscosity
  regimes, larger grids, longer horizons, different seeds, or stronger training
  budgets.
