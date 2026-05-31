# Technical Q&A

## Claim Boundaries

### Can we say SM-FNO outperforms FNO or Transformer?

No. The repository currently contains protocol-scale smoke and repeated-seed
runs. These are useful for checking data generation, model wiring, rollout
evaluation, aggregation, plotting, and reporting, but they are not final
benchmark experiments.

### What can we safely claim?

The repository supports config-driven experiments for Heat1D, Burgers1D, and
low-resolution Navier-Stokes2D with FNO, SM-FNO, and Transformer baselines. It
also includes repeated-seed execution, rollout-20 validation configs,
aggregation, cost-reporting utilities, and SM-FNO2D v2 as a separate stabilized
variant.

## Research Question

### What is the main research question?

Can long-horizon PDE dynamics be forecast effectively without temporal
self-attention by combining spectral spatial operators with state space
temporal memory?

### Why is this interesting?

Autoregressive PDE rollout can require many temporal steps. Temporal
self-attention can be expressive, but pairwise sequence interactions can become
costly as sequence length grows. SSM-style memory offers a linear-time temporal
alternative worth testing under fair protocols.

## Mathematical Framing

### What is the forecasting task?

Given an input window of field states,
`u(t_0), ..., u(t_k)`, predict one or more future states
`u(t_{k+1}), ..., u(t_{k+H})`.

### Why report rollout metrics?

One-step error measures local prediction quality. Rollout error measures what
happens when predictions are fed back into the model, where errors can
compound over many steps.

### What metrics are currently logged?

The current pipeline logs MSE, relative L2, rollout relative L2, per-timestep
rollout relative L2, inference timing fields, and parameter-count fields.

## Model Components

### What does FNO contribute?

FNO layers model spatial structure by applying learned weights to selected
Fourier modes. This provides a global spatial receptive field without using
attention over all grid points.

### What does the SSM contribute?

The SSM contributes temporal memory through a recurrence such as
`h_t = A h_{t-1} + B x_t`. In this project it is used to carry latent temporal
state across input windows and rollouts.

### What does the Transformer baseline contribute?

The Transformer baseline provides an attention-based temporal reference. It is
included so the attention-free hypothesis can eventually be tested against an
expressive baseline under the same data, budget, and rollout protocol.

### What changed in SM-FNO2D v2?

SM-FNO2D v2 adds a stable gated temporal memory path. The transition is
parameterized so recurrent decay stays in `(0, 1)`, and the memory output is
combined with a residual current-input path through a learned gate. It is kept
as a separate model group from the original SM-FNO2D.

## Data and Solvers

### Are the PDE solvers production grade?

No. The data generators are CPU-friendly protocol fixtures. They are useful for
testing experiment mechanics and shape handling, but not for final numerical
claims.

### What datasets are currently covered?

- Heat1D: smooth dissipative dynamics.
- Burgers1D: nonlinear viscous advection-diffusion dynamics.
- Navier-Stokes2D: low-resolution periodic vorticity forecasting.

### Why use small datasets?

The current milestones prioritize reproducible local verification. Larger
grids, longer horizons, and more seeds are future work.

## Protocol and Evidence

### How is fairness handled?

The protocol uses shared data splits, `input_steps`, `pred_steps`,
`rollout_steps`, training settings where appropriate, repeated seeds, and
shared metrics. The configs are plain YAML and the scripts are config-driven.

### What does repeated-seed support show?

It confirms the workflow can run and aggregate multiple seeds. It does not prove
statistical significance because the current seed count is intentionally small.

### What does cost-efficiency analysis currently mean?

It means the pipeline logs parameter counts and local CPU timing fields, then
aggregates them by dataset, model, horizon, and seed. These are local
protocol-scale measurements, not final efficiency claims.

## Review Fixes and Stability

### What was the FNO2D small-grid issue?

For small or odd 2D grids, retained top and bottom y-frequency rows could
overlap in spectral convolution. The fix clamps retained top and bottom rows so
the bottom assignment cannot overwrite the top assignment.

### Why not reject small grids instead?

Small grids are useful for CPU-friendly smoke tests. Clamping lets valid small
configs run while making retained spectral rows explicit and non-overlapping.

### What stability guard was added for Burgers1D?

The Burgers1D generator enforces a combined explicit stability condition:
`cfl + 2 * diffusion_number <= stability_safety`.

## Figure and Report Paths

### Where are result diagnostics?

- Prediction figures: `results/figures/*_prediction.png`
- Rollout diagnostics: `results/figures/*_rollout_relative_l2.png`
- Training curves: `results/figures/*_training_loss.png`
- Cost plots: `results/figures/m9_cost_efficiency/*.png` and
  `results/figures/m11_navier_stokes2d_v2_cost/*.png`
- Aggregate tables: `results/tables/*_aggregate*.md`

These files are generated artifacts and should remain out of git.

## Future Work

### What would be needed for real benchmark claims?

- Larger and better validated PDE datasets.
- More seeds and confidence intervals.
- Matched or justified training budgets.
- Stronger baselines and tuning rules.
- Memory and FLOP measurements, not just local timing.
- Domain diagnostics such as energy, spectrum, and conservation checks.
- Clear frozen configs, metrics, plots, and verification commands.
