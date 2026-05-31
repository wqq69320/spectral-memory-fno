# Final Results Caveats

## Claim Policy

Do not claim that SM-FNO, SM-FNO2D, SM-FNO2D v2, FNO, or Transformer is better,
faster, more accurate, more efficient, or more cost-effective based on the
current smoke and protocol-scale outputs.

Allowed wording:

- "The repository supports..."
- "The pipeline records..."
- "The smoke run produced..."
- "The protocol-scale run validates the execution path..."
- "The current artifact is a sanity check..."

Avoided wording:

- "SM-FNO outperforms..."
- "SM-FNO is more efficient..."
- "Transformer is worse..."
- "FNO is the best..."
- "These results prove..."

## What Current Results Prove

- Config-driven data generation works for Heat1D, Burgers1D, and
  Navier-Stokes2D protocol fixtures.
- FNO, SM-FNO, Transformer, and SM-FNO2D v2 model paths can be trained and
  evaluated through the shared scripts.
- Autoregressive rollout metrics and per-step rollout diagnostics are saved.
- Repeated-seed execution and aggregation are implemented.
- Cost-reporting utilities can aggregate parameter counts and local CPU timing
  fields.
- Generated artifacts are written under ignored local output paths.

## What Current Results Do Not Prove

- They do not prove model superiority.
- They do not prove final accuracy on benchmark PDE datasets.
- They do not prove final cost-efficiency.
- They do not prove robustness to larger grids, longer rollouts, or harder
  dynamics.
- They do not prove numerical solver fidelity.
- They do not provide enough seeds for statistical conclusions.

## Dataset Caveats

### Heat1D

Heat1D is useful for smooth dissipative dynamics and early pipeline validation.
It is not sufficient to validate difficult nonlinear long-horizon behavior.

### Burgers1D

Burgers1D adds nonlinear advection-diffusion dynamics. The generator includes a
combined explicit stability guard, but it remains a simple CPU-friendly
protocol fixture.

### Navier-Stokes2D

Navier-Stokes2D is a low-resolution periodic vorticity fixture. It is useful
for 2D shape handling and workflow validation, but it is not a final CFD
benchmark.

## Model Caveats

### FNO

FNO is a spatial spectral baseline. Current small configs are smoke-friendly
and should not be treated as tuned final baselines.

### SM-FNO

SM-FNO combines spectral spatial modeling with SSM temporal memory. Current
implementations validate the architecture path but are not final tuned models.

### Transformer

The Transformer baseline is included as an attention-based reference. Current
configs are CPU-friendly and do not establish final attention-baseline quality.

### SM-FNO2D v2

SM-FNO2D v2 adds stable gated temporal memory and is reported separately from
SM-FNO2D v1. Current v2 artifacts show that the implementation path runs; they
do not establish final model quality.

## Metric Caveats

- MSE and relative L2 are useful but incomplete.
- Rollout relative L2 exposes error accumulation but does not describe all
  physical behavior.
- Local CPU timing is sensitive to machine load and implementation details.
- Parameter count is a rough proxy and does not replace FLOPs, memory, or
  hardware-aware profiling.
- Current aggregate tables may include single-seed groups alongside repeated
  seed groups; read the `Run Type`, `Count`, and `Seeds` fields before
  interpreting a row.

## Figure Caveats

- Prediction figures are qualitative diagnostics.
- Rollout-relative-L2 figures are sanity diagnostics.
- Cost plots are reporting-mechanism diagnostics.
- Training-loss curves should not be used as final evidence of model quality.
- Generated figures should remain out of git and be reproduced from configs
  when needed.

## Required Evidence Before Benchmark Claims

Before making any comparative claim, require:

- Frozen data-generation configs.
- Frozen model and training configs.
- Clearly documented train/validation/test splits.
- Matched or justified training budgets.
- More repeated seeds and uncertainty reporting.
- Larger and validated datasets.
- Saved metrics, plots, checkpoints, and command logs.
- Cost measurements beyond local wall-clock timing.
- Domain-specific diagnostics such as energy and spectrum errors.
- A written verification report that states exactly what claim the evidence
  supports.

## Safe Final Forum Summary

The current project should be presented as a reproducible research scaffold for
testing an attention-free PDE forecasting hypothesis. It has strong protocol
coverage for its stage, but its current metrics are sanity checks only.
