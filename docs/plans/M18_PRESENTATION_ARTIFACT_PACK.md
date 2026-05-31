# M18 Presentation Artifact Pack Plan

## Objective

Create a polished presentation-ready artifact pack for the final
`spectral-memory-fno` forum talk. The pack should focus on the dynamic
Navier-Stokes2D fixture and optimized SM-FNO2D v3 diagnostic from M17.

The goal is communication quality, not new benchmark claims.

## Story Scope

- SM-FNO2D v2 showed one-step promise but was not rollout-stable enough.
- SM-FNO2D v3 used an FNO residual base plus gated SSM correction to reduce
  visible rollout collapse.
- Persistence checks showed that low rollout error alone was not enough.
- A more dynamic Navier-Stokes2D fixture made persistence harder.
- Optimized SM-FNO2D v3 used FNO-base initialization plus delta-aware and
  persistence-aware rollout training to achieve positive full-test dynamic skill
  on the diagnostic fixture.

## Required Outputs

Generate outputs under ignored paths:

- `results/figures/m18_presentation/`
- `results/tables/m18_presentation/`

Required artifact types:

- Final model comparison table.
- Rollout relative L2 comparison plot.
- Persistence-normalized skill plot.
- Held-out temporal variation plot.
- Delta-prediction error plot.
- Compact cost-vs-performance figure.
- Static true / predicted / absolute-error panels at steps 1, 18, and 36.
- Step-36 model comparison panel across true, persistence, FNO2D, and optimized
  SM-FNO2D v3.
- GIFs for optimized SM-FNO2D v3, FNO2D, persistence, and a side-by-side model
  comparison.
- 3D-style surface renders for optimized SM-FNO2D v3 at representative steps.
- `ARTIFACT_INDEX.md` for slide-making.

## Implementation Plan

1. Add a reproducible script that reads the M17 full-test rollout artifacts.
2. Recompute presentation metrics directly from full held-out rollouts.
3. Use dark-theme-compatible figure styling with legible titles and labels.
4. Write Markdown, CSV, PNG table, summary card, GIFs, and figure manifest files.
5. Keep all generated artifacts under ignored `results/` paths.
6. Document selected hero artifacts, commands, limitations, and claim boundaries.

## Verification Commands

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/create_m18_presentation_artifacts.py
git diff --check
git check-ignore -v \
  results/figures/m18_presentation/m18_v3_rollout.gif \
  results/tables/m18_presentation/ARTIFACT_INDEX.md
```

## Done Criteria

- M18 docs are created.
- Final presentation-ready figures, tables, and GIFs are generated.
- `results/tables/m18_presentation/ARTIFACT_INDEX.md` exists.
- Tests pass.
- Generated artifacts remain ignored.
- No unsupported benchmark claims are added.
