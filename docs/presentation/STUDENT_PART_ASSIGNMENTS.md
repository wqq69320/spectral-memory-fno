# Student Part Assignments

## Purpose

Split a 15 minute forum presentation into clear student-owned sections. Each
part should make the research scaffold understandable without making benchmark
claims from smoke runs.

## Assignment Summary

| Student | Time | Section | Responsibility | Primary References |
| --- | ---: | --- | --- | --- |
| Student A | 3 min | Motivation and mathematical framing | Introduce the research question, PDE rollout, one-step vs autoregressive error, and why fair protocols matter. | `docs/00_project_overview.md`, `docs/01_mathematical_foundation.md`, `docs/05_experiment_protocol.md` |
| Student B | 3 min | FNO, SSM, and SM-FNO architecture | Explain spectral spatial modeling, state space temporal memory, and how SM-FNO combines them. | `docs/02_fourier_neural_operator.md`, `docs/03_state_space_model.md`, `src/sm_fno/models/` |
| Student C | 3 min | Baselines and datasets | Cover FNO, SM-FNO, Transformer, Heat1D, Burgers1D, Navier-Stokes2D, and the role of smoke configs. | `configs/experiment/`, `docs/reports/M8_VERIFICATION_REPORT.md`, `docs/reports/M10_VERIFICATION_REPORT.md` |
| Student D | 3 min | M11 extension and cost analysis | Explain SM-FNO2D v2, small-grid FNO2D fix, repeated seeds, horizon sweeps, and cost-reporting artifacts. | `docs/plans/M11_SMFNO2D_IMPROVEMENT.md`, `docs/decision_logs/M11_DECISION_LOG.md`, `docs/reports/M11_VERIFICATION_REPORT.md` |
| Student E | 3 min | Caveats, future work, and Q&A lead | State limitations, claim boundaries, next experiments, and field technical questions. | `docs/reports/FINAL_RESULTS_CAVEATS.md`, `docs/presentation/TECHNICAL_QA.md` |

If there are fewer than five presenters, combine Student C and Student D, then
have one person handle caveats and Q&A.

## Detailed Responsibilities

### Student A: Motivation and Framing

- Start with the question: can long-horizon PDE dynamics be forecast without
  temporal attention by combining FNO spatial operators with SSM memory?
- Define the forecasting setup:
  `u(t_0), ..., u(t_k) -> u(t_{k+1}), ..., u(t_{k+H})`.
- Explain why rollout evaluation matters: prediction errors can compound when
  model outputs feed later inputs.
- Establish the claim boundary: current results are smoke/protocol sanity
  checks only.

### Student B: Architecture

- Explain FNO as FFT, learned low-mode spectral multiplication, inverse FFT,
  and pointwise channel mixing.
- Explain SSM memory with the recurrence
  `h_t = A h_{t-1} + B x_t`, `y_t = C h_t + D x_t`.
- Explain the SM-FNO composition: spatial operator for each field state plus
  temporal memory over latent states.
- Mention that temporal attention remains as a baseline, not as something the
  project dismisses.

### Student C: Protocol and Data

- Cover Heat1D as smooth dissipative dynamics.
- Cover Burgers1D as nonlinear advection-diffusion dynamics.
- Cover Navier-Stokes2D as a low-resolution periodic vorticity protocol
  fixture.
- Explain shared protocol dimensions: data split, `input_steps`, `pred_steps`,
  `rollout_steps`, training settings, seeds, and metrics.
- Use figures only as examples of generated diagnostics.

### Student D: M11 and Cost Tooling

- Explain the reviewed FNO2D small-grid spectral slicing issue and the fix:
  retained top/bottom y modes are clamped to non-overlapping rows.
- Explain SM-FNO2D v2 as a separate stabilized variant with gated residual
  temporal memory.
- Explain horizon sweeps at 5, 10, and 20 rollout steps.
- Explain cost reporting as local CPU timing and parameter-count logging, not
  final cost-efficiency evidence.

### Student E: Caveats and Q&A

- State what is shown: code paths run, tests pass, artifacts are generated,
  and protocol tools exist.
- State what is not shown: model superiority, generalization to larger PDE
  systems, final efficiency, and production solver accuracy.
- Keep answers concise and return to the fair-comparison standard.

## Shared Language Rules

- Use "supports", "implements", "validates the pipeline", and "records sanity
  metrics".
- Avoid "wins", "beats", "best", "superior", "state of the art",
  "benchmark result", and "final result".
- If showing a metric table, label it "smoke sanity output" and explain that
  more seeds, larger grids, and stronger diagnostics are required before
  conclusions.

## Handoff Checklist

- Each student should know which slide numbers they own.
- Each student should point to at least one repository path as evidence.
- Each student should avoid interpreting smoke metrics as rankings.
- The final presenter should close with future work rather than a performance
  claim.
