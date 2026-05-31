# M12 Forum Wrap-up Presentation Package Plan

## Objective

Create presentation-ready documentation for a final forum discussion of the
SM-FNO research scaffold. The package should explain the research question,
mathematical framing, model components, experiment protocol, Navier-Stokes2D
extension, SM-FNO2D v2 improvement, cost-efficiency tooling, limitations, and
future work without making benchmark claims.

## Scope

- Add a 15 minute talk outline.
- Add student part assignments for a multi-presenter forum talk.
- Add a technical Q&A document.
- Add a figure-selection guide that references generated artifact paths.
- Add a final forum report draft.
- Add final results caveats and claim boundaries.
- Update README Current Status with the M12 wrap-up package.
- Verify that referenced generated artifact paths exist locally and remain
  ignored by git.

## Done Criteria

- [x] `docs/presentation/15MIN_TALK_OUTLINE.md` exists and covers talk timing,
  slide purpose, speaker notes, and backup evidence paths.
- [x] `docs/presentation/STUDENT_PART_ASSIGNMENTS.md` exists and assigns
  presentation responsibilities.
- [x] `docs/presentation/TECHNICAL_QA.md` exists and prepares answers for
  research, model, protocol, cost, and caveat questions.
- [x] `docs/presentation/FIGURE_SELECTION_GUIDE.md` exists and lists
  presentation-safe figure/table paths.
- [x] `docs/reports/FINAL_FORUM_REPORT_DRAFT.md` exists and summarizes the
  project narrative.
- [x] `docs/reports/FINAL_RESULTS_CAVEATS.md` exists and documents claim
  boundaries.
- [x] README Current Status mentions M12.
- [x] Generated artifact paths referenced by the package were checked.
- [x] The docs avoid unsupported performance claims and model rankings.

## Required Verification

```bash
test -f docs/presentation/15MIN_TALK_OUTLINE.md
test -f docs/presentation/STUDENT_PART_ASSIGNMENTS.md
test -f docs/presentation/TECHNICAL_QA.md
test -f docs/presentation/FIGURE_SELECTION_GUIDE.md
test -f docs/reports/FINAL_FORUM_REPORT_DRAFT.md
test -f docs/reports/FINAL_RESULTS_CAVEATS.md
test -f results/figures/heat1d_sm_fno_smoke_prediction.png
test -f results/figures/burgers1d_sm_fno_smoke_prediction.png
test -f results/figures/navier_stokes2d_sm_fno_v2_smoke_prediction.png
test -f results/figures/m11_navier_stokes2d_v2_cost/rollout_seconds_per_step_by_horizon.png
git check-ignore -v results/figures/heat1d_sm_fno_smoke_prediction.png results/tables/m11_navier_stokes2d_v2_cost_report.md
git diff --check
```

## Non-Goals

- Do not generate or commit new figures.
- Do not rerun experiments unless documentation verification requires it.
- Do not make benchmark, ranking, or superiority claims.
- Do not add new model or data code.
