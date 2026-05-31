# Final Forum Report Draft

## Title

Can We Forecast Long-Horizon PDE Dynamics Without Attention? Combining Fourier
Neural Operators with State Space Memory

## Abstract

This project studies whether Fourier Neural Operator spatial modeling can be
combined with State Space Model temporal memory for long-horizon PDE
forecasting without relying on temporal self-attention. The current repository
implements a reproducible research scaffold across Heat1D, Burgers1D, and
low-resolution Navier-Stokes2D. It includes FNO, SM-FNO, and Transformer
baselines, repeated-seed execution, rollout metrics, aggregation, plotting,
and protocol-scale cost-reporting utilities. A separate SM-FNO2D v2 path adds
stable gated temporal memory for Navier-Stokes2D experiments. Current metrics
are smoke and protocol sanity outputs only; they do not support benchmark
claims or model rankings.

## Research Question

Can long-horizon PDE dynamics be forecast effectively without temporal
self-attention by combining spectral spatial modeling with state space temporal
memory?

The working hypothesis is that FNO-style spatial operators and SSM-style
temporal recurrence may provide a useful attention-free route for PDE rollout,
especially when long temporal horizons make attention cost an important
consideration. This remains a hypothesis to be tested under larger and more
rigorous experiments.

## Mathematical Framing

PDE forecasting approximates an evolution operator over spatial fields:

```text
u(t_0), ..., u(t_k) -> u(t_{k+1}), ..., u(t_{k+H})
```

where `u(t)` is a spatial field and `H` is the forecast horizon. Two evaluation
views matter:

- One-step prediction: local next-state accuracy.
- Autoregressive rollout: repeated prediction where model outputs become later
  inputs.

Rollout evaluation is important because errors can compound over time even
when one-step metrics look reasonable.

## Model Framing

### Fourier Neural Operator

The FNO baseline models spatial fields by applying learned weights to selected
Fourier modes:

```text
field -> FFT -> learned spectral weights -> inverse FFT -> updated field
```

This gives a global spatial receptive field without attention over all grid
points.

### State Space Memory

The SSM component maintains temporal memory through a recurrence:

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t + D x_t
```

This style of temporal update scales linearly with sequence length under
recurrent execution, making it a plausible alternative to temporal attention
for long rollouts.

### SM-FNO

SM-FNO combines FNO-style spatial processing with SSM-style temporal memory.
The intent is to separate spatial structure from temporal memory while keeping
the experiment pipeline comparable with FNO and Transformer baselines.

### Attention Baseline

The Transformer baseline remains important because attention is an expressive
temporal mechanism. It provides a reference for testing whether the
attention-free approach is useful under the same data split, rollout horizon,
training budget, and metrics.

## Experiment Scaffold

The repository currently supports:

- Heat1D smoke and rollout configs for MLP, FNO, SM-FNO, and Transformer
  paths.
- Burgers1D smoke and rollout configs for FNO, SM-FNO, and Transformer paths.
- Navier-Stokes2D smoke and rollout configs for FNO2D, SM-FNO2D,
  SM-FNO2D v2, and Transformer2D paths.
- Repeated-seed execution through `scripts/run_repeated_seeds.py`.
- Metric aggregation through `scripts/aggregate_metrics.py`.
- Horizon sweeps and cost aggregation through the M9/M10/M11 cost utilities.

All experiment execution is config-driven through plain YAML.

## Navier-Stokes2D Extension

The M10 milestone added a low-resolution periodic Navier-Stokes2D vorticity
forecasting path. It introduced 2D data generation, 2D model classes, 5D
rollout shape handling, plotting support, and Navier-Stokes2D smoke and
rollout-20 configs.

The Navier-Stokes2D generator is a protocol fixture, not a production CFD
solver. It is useful for testing 2D shape handling, plotting, rollout metrics,
and model wiring before moving to larger validated datasets.

## SM-FNO2D v2 Improvement

The M11 milestone added two related improvements:

- A FNO2D spectral-convolution fix for small or odd grids: retained top and
  bottom y-frequency rows are clamped so they do not overlap.
- A separate `SpectralMemoryFNO2DV2` path with stable gated temporal memory and
  residual spatial blocks.

SM-FNO2D v2 is intentionally separate from the original SM-FNO2D model so the
two variants can be reported as different model groups. In the current small
Navier-Stokes2D sanity runs, v2 changed the recorded rollout behavior relative
to v1, but these observations are implementation signals only. They are not
benchmark evidence.

## Cost-Efficiency Analysis

The cost-efficiency tooling records:

- Parameter counts.
- One-step inference timing.
- Rollout inference timing.
- Rollout seconds per step.
- Rollout seconds per example per step.
- Horizon-sweep aggregates at horizons 5, 10, and 20.

M9 covers Heat1D and Burgers1D. M10 covers Navier-Stokes2D. M11 adds
SM-FNO2D v2 as a separate Navier-Stokes2D model group. The timing values are
local CPU measurements and should be treated as reporting-mechanism evidence,
not final efficiency results.

## Current Evidence

The current evidence shows that the research scaffold runs end to end:

- Data generation, training, evaluation, plotting, repeated seeds, aggregation,
  and horizon sweeps execute locally.
- Tests cover imports, model shapes, rollout handling, stability guards, and
  selected config expansion behavior.
- Generated figures and tables exist under ignored output directories.
- Verification reports document commands and artifact paths.

Current evidence does not prove model superiority, final accuracy, final
efficiency, or generalization to larger physical systems.

## Presentation Artifact Paths

Useful generated diagnostics include:

- Heat1D prediction:
  `results/figures/heat1d_sm_fno_smoke_prediction.png`
- Burgers1D prediction:
  `results/figures/burgers1d_sm_fno_smoke_prediction.png`
- Navier-Stokes2D v2 prediction:
  `results/figures/navier_stokes2d_sm_fno_v2_smoke_prediction.png`
- Navier-Stokes2D v2 rollout-20 diagnostic:
  `results/figures/navier_stokes2d_sm_fno_v2_rollout20_rollout_relative_l2.png`
- M9 cost report:
  `results/tables/m9_cost_efficiency_report.md`
- M11 cost report:
  `results/tables/m11_navier_stokes2d_v2_cost_report.md`

These are generated artifacts and should remain out of git.

## Limitations

- Datasets are small CPU-friendly protocol fixtures.
- Navier-Stokes2D is low resolution and not a validated CFD benchmark.
- Current repeated-seed counts are small.
- Hyperparameters are smoke-friendly rather than final tuned settings.
- Timing measurements are local CPU wall-clock values.
- Cost analysis does not yet include FLOPs or memory profiling.
- Domain diagnostics such as energy error and spectrum error remain future
  work.

## Future Work

- Run larger, numerically validated datasets.
- Increase seed counts and report uncertainty with a clear statistical plan.
- Add memory and FLOP measurement.
- Add physical diagnostics such as energy, spectrum, and conservation error.
- Tune model families under a documented fair-comparison protocol.
- Freeze final benchmark configs before making any comparative claims.
- Prepare a public release report only after configs, metrics, plots, and
  verification commands support the claims being made.

## Forum Takeaway

The project has reached a presentation-ready scaffold stage: it can explain a
clear attention-free PDE forecasting hypothesis, demonstrate a reproducible
multi-dataset protocol, and show generated diagnostic artifacts. The correct
conclusion is that the infrastructure is ready for more rigorous experiments,
not that any model has already won.
