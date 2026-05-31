# M18 Verification Report

M18 artifacts are presentation diagnostics only. They are not benchmark claims,
model rankings, or evidence of general Navier-Stokes forecasting performance.

## Objective

Create a polished presentation-ready artifact pack for the final forum talk,
focused on the dynamic Navier-Stokes2D fixture and optimized SM-FNO2D v3 from
M17.

## Implementation Summary

- Added `scripts/create_m18_presentation_artifacts.py` as a reproducible entry
  point for generating the final presentation pack from M17 full-test rollout
  artifacts.
- Generated dark-theme-compatible summary plots, static rollout panels, model
  comparison panels, 3D-style surface renders, GIFs, a final takeaway card, and
  Markdown/CSV/PNG tables.
- Added 3D-style animated rollout GIFs for the true target, persistence, FNO2D,
  optimized SM-FNO2D v3, and a side-by-side true/persistence/FNO/v3 comparison.
- The animated 3D-style outputs are surface renderings of 2D vorticity fields,
  not true 3D fluid forecasts.
- Added `configs/experiment/navier_stokes2d_dynamic_transformer_diagnostic.yaml`
  for a CPU-feasible Transformer2D temporal-attention diagnostic on the dynamic
  fixture.
- Regenerated M18 artifacts so the final comparison includes persistence,
  FNO2D, Transformer2D, and optimized SM-FNO2D v3.
- Transformer2D is a temporal-attention baseline applied per spatial grid cell,
  not a full spatiotemporal Transformer over all space-time tokens.
- Wrote `results/tables/m18_presentation/ARTIFACT_INDEX.md` for fast
  slide-making.
- Selected held-out sample `4` automatically because it has the largest step-36
  persistence error among the held-out trajectories, making temporal change
  visually clear.
- MP4 export was attempted with `--write-mp4`, but no ffmpeg writer was
  available in this environment. GIFs were generated successfully.

## Commands Run

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q` | Pass | 54 tests passed. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_dynamic_transformer_diagnostic.yaml` | Pass | Trained the temporal-attention baseline. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_dynamic_transformer_diagnostic.yaml` | Pass | Saved full-test Transformer2D rollouts. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_dynamic_transformer_diagnostic.yaml` | Pass | Saved Transformer2D plots. |
| `PYTHONPATH=src python3 scripts/run_dynamic_skill_diagnostics.py ...` | Pass | Wrote M18 attention dynamic-skill diagnostics. |
| `PYTHONPATH=src python3 scripts/create_m18_presentation_artifacts.py --write-mp4` | Pass | Generated M18 artifacts with Transformer2D; MP4 skipped because ffmpeg writer is unavailable. |
| `git diff --check` | Pass | No whitespace errors. |
| `git check-ignore -v ... representative M18 artifacts ...` | Pass | M18 outputs remain ignored. |

## Diagnostic Metrics Used In Presentation

| Model | Rel L2 Mean | Rel L2 Step 36 | Skill Mean | Skill Step 36 | Delta Error Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Persistence | 0.127381 | 0.238863 | 0.000000 | 0.000000 | 1.000000 |
| FNO2D | 0.139531 | 0.226624 | -0.461319 | 0.051239 | 1.528133 |
| Transformer2D | 0.219706 | 0.381732 | -0.974676 | -0.598120 | 2.028151 |
| SM-FNO2D v3 optimized | 0.024568 | 0.038854 | 0.743001 | 0.837336 | 0.263177 |

## Hero Artifacts

- Architecture asset: `docs/assets/sm_fno2d_structure_dark.svg`
- Rollout stability plot:
  `results/figures/m18_presentation/m18_rollout_relative_l2_comparison.png`
- Persistence comparison plot:
  `results/figures/m18_presentation/m18_persistence_normalized_skill.png`
- Dynamic-skill plot:
  `results/figures/m18_presentation/m18_delta_prediction_error.png`
- Best GIF:
  `results/figures/m18_presentation/m18_true_persistence_fno_transformer_v3_comparison.gif`
- Best single 3D-style GIF:
  `results/figures/m18_presentation/3d_style/m18_3d_v3_rollout.gif`
- Best side-by-side 3D-style GIF:
  `results/figures/m18_presentation/3d_style/m18_3d_true_persistence_fno_transformer_v3_comparison.gif`
- Best Attention-vs-SM-FNO figure:
  `results/figures/m18_presentation/m18_step36_model_comparison.png`
- Best Transformer2D rollout GIF:
  `results/figures/m18_presentation/m18_transformer_rollout.gif`
- Best table including Transformer2D:
  `results/tables/m18_presentation/m18_final_model_comparison.md`
- 3D-style figure:
  `results/figures/m18_presentation/m18_v3_surface_step36.png`
- Final takeaway slide asset:
  `results/figures/m18_presentation/m18_presentation_summary_card.png`

## Generated Artifact Groups

Tables:

- `results/tables/m18_presentation/ARTIFACT_INDEX.md`
- `results/tables/m18_presentation/m18_artifact_manifest.json`
- `results/tables/m18_presentation/m18_final_model_comparison.csv`
- `results/tables/m18_presentation/m18_final_model_comparison.md`
- `results/tables/m18_presentation/m18_final_model_comparison_table.png`
- `results/tables/m18_presentation/m18_attention_dynamic_skill.json`
- `results/tables/m18_presentation/m18_attention_dynamic_skill.md`

Figures:

- `results/figures/m18_presentation/m18_rollout_relative_l2_comparison.png`
- `results/figures/m18_presentation/m18_persistence_normalized_skill.png`
- `results/figures/m18_presentation/m18_heldout_temporal_variation.png`
- `results/figures/m18_presentation/m18_delta_prediction_error.png`
- `results/figures/m18_presentation/m18_cost_vs_diagnostic_error.png`
- `results/figures/m18_presentation/m18_step36_model_comparison.png`
- `results/figures/m18_presentation/m18_presentation_summary_card.png`
- `results/figures/m18_presentation/m18_transformer_step01_panels.png`
- `results/figures/m18_presentation/m18_transformer_step18_panels.png`
- `results/figures/m18_presentation/m18_transformer_step36_panels.png`
- `results/figures/m18_presentation/attention_dynamic_skill/m18_attention_rollout_vs_persistence.png`
- `results/figures/m18_presentation/attention_dynamic_skill/m18_attention_persistence_skill.png`
- `results/figures/m18_presentation/attention_dynamic_skill/m18_attention_delta_prediction_error.png`
- `results/figures/m18_presentation/m18_fno_step01_panels.png`
- `results/figures/m18_presentation/m18_fno_step18_panels.png`
- `results/figures/m18_presentation/m18_fno_step36_panels.png`
- `results/figures/m18_presentation/m18_persistence_step01_panels.png`
- `results/figures/m18_presentation/m18_persistence_step18_panels.png`
- `results/figures/m18_presentation/m18_persistence_step36_panels.png`
- `results/figures/m18_presentation/m18_v3_step01_panels.png`
- `results/figures/m18_presentation/m18_v3_step18_panels.png`
- `results/figures/m18_presentation/m18_v3_step36_panels.png`

GIFs:

- `results/figures/m18_presentation/m18_v3_rollout.gif`
- `results/figures/m18_presentation/m18_fno_rollout.gif`
- `results/figures/m18_presentation/m18_transformer_rollout.gif`
- `results/figures/m18_presentation/m18_persistence_rollout.gif`
- `results/figures/m18_presentation/m18_true_persistence_fno_transformer_v3_comparison.gif`
- `results/figures/m18_presentation/3d_style/m18_3d_true_rollout.gif`
- `results/figures/m18_presentation/3d_style/m18_3d_v3_rollout.gif`
- `results/figures/m18_presentation/3d_style/m18_3d_fno_rollout.gif`
- `results/figures/m18_presentation/3d_style/m18_3d_transformer_rollout.gif`
- `results/figures/m18_presentation/3d_style/m18_3d_persistence_rollout.gif`
- `results/figures/m18_presentation/3d_style/m18_3d_true_persistence_fno_transformer_v3_comparison.gif`

3D-style visualizations:

- `results/figures/m18_presentation/m18_v3_surface_step01.png`
- `results/figures/m18_presentation/m18_v3_surface_step18.png`
- `results/figures/m18_presentation/m18_v3_surface_step36.png`

## Caveats And Claim Boundaries

- The M18 pack visualizes one synthetic dynamic fixture and one optimized
  diagnostic run.
- 3D-style animations show 2D vorticity fields as surfaces; they do not imply
  true 3D Navier-Stokes forecasting.
- Positive dynamic skill is a diagnostic result on this fixture, not a broad
  benchmark claim.
- Persistence, FNO2D, Transformer2D, and optimized SM-FNO2D v3 are included as
  final talk anchors for the dynamic fixture diagnostic.
- Transformer2D is now included for the dynamic fixture, but it is a small
  temporal-attention baseline applied per spatial grid cell. It is not a full
  spatiotemporal attention baseline and should not be used for broad benchmark
  claims.

## Deferred Items

- MP4 outputs were attempted through Matplotlib's ffmpeg writer but skipped
  because no ffmpeg writer is available in this environment.
- Full spatiotemporal Transformer attention remains deferred.
- Direct slide-deck export was deferred; M18 produces reusable slide assets
  instead.
