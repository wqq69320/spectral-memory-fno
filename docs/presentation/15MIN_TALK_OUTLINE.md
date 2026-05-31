# 15 Minute Talk Outline

## Talk Goal

Explain the research question, model idea, protocol progress, and current
limitations of SM-FNO without presenting smoke metrics as final benchmark
results.

Recommended title:

```text
Can We Forecast Long-Horizon PDE Dynamics Without Attention?
Combining Fourier Neural Operators with State Space Memory
```

## Core Message

This project asks whether spectral spatial modeling from Fourier Neural
Operators can be paired with state space temporal memory to support
long-horizon PDE forecasting without relying on temporal self-attention. The
current repository is a reproducible research scaffold across Heat1D,
Burgers1D, and low-resolution Navier-Stokes2D. It demonstrates protocol
coverage, model wiring, rollout evaluation, repeated-seed tooling, and
cost-reporting mechanics, but it does not yet support benchmark claims.

## Slide Plan

| Time | Slide | Purpose | Suggested Visual |
| ---: | --- | --- | --- |
| 0:00-1:00 | Title and question | State the problem and hypothesis. | No generated result figure needed. |
| 1:00-2:00 | Why PDE rollout is hard | Explain autoregressive error compounding. | `results/figures/heat1d_fno_rollout20_rollout_relative_l2.png` as an example rollout diagnostic. |
| 2:00-3:15 | Mathematical framing | Define `u(t_0:t_k) -> u(t_{k+1}:t_{k+H})`, one-step error, and rollout error. | Simple equation slide; use text, not a metric ranking. |
| 3:15-4:30 | Fourier Neural Operator | Explain FFT, retained modes, learned spectral weights, inverse FFT. | Optional architecture diagram in slides; avoid using a performance plot. |
| 4:30-5:45 | State space memory | Explain recurrent memory `h_t = A h_{t-1} + B x_t` and linear temporal scaling. | Simple recurrence diagram. |
| 5:45-7:00 | Attention baseline | Explain why Transformer baselines are included and what cost question they test. | Cost scaling note: attention over time has pairwise sequence interactions. |
| 7:00-8:30 | Experiment protocol | Same splits, input steps, prediction steps, rollout steps, seeds, optimizer budget where appropriate. | `results/tables/aggregate_metrics.md` or `results/tables/navier_stokes2d_m11_aggregate_metrics.md` as backup evidence, not a slide ranking. |
| 8:30-10:00 | PDE coverage | Heat1D, Burgers1D, Navier-Stokes2D progression. | `results/figures/heat1d_sm_fno_smoke_prediction.png`, `results/figures/burgers1d_sm_fno_smoke_prediction.png`, `results/figures/navier_stokes2d_sm_fno_v2_smoke_prediction.png`. |
| 10:00-11:15 | SM-FNO2D v2 | Explain stable gated temporal memory, residual path, and why v2 is separate from v1. | `results/figures/navier_stokes2d_sm_fno_v2_rollout20_rollout_relative_l2.png` as a sanity diagnostic. |
| 11:15-12:30 | Cost-efficiency tooling | Explain parameters, rollout seconds per step, and horizon sweeps as reporting mechanics. | `results/figures/m11_navier_stokes2d_v2_cost/rollout_seconds_per_step_by_horizon.png` or `results/figures/m11_navier_stokes2d_v2_cost/rollout_error_vs_seconds_per_step.png`. |
| 12:30-13:45 | What the current evidence means | Current runs validate implementation and reporting. They do not rank models. | Use a caveat slide with paths to generated reports. |
| 13:45-14:45 | Limitations and next work | Larger datasets, more seeds, stronger diagnostics, memory/FLOP reporting, final benchmarks. | No result plot needed. |
| 14:45-15:00 | Close | Restate question and next milestone. | One-line takeaway. |

## Speaker Notes

- Use "protocol-scale sanity check" for all current metrics.
- Say "the pipeline now supports..." rather than "the model achieves...".
- When referencing SM-FNO2D v2, describe the implementation change: stable
  transition parameterization and gated residual temporal memory.
- Do not say SM-FNO is better, faster, more accurate, or more efficient than a
  baseline. The current evidence is not benchmark-scale.
- If someone asks for results, point to saved smoke artifacts and explain that
  they show reproducibility mechanics, not final scientific conclusions.

## Minimal Slide Count

If the presentation needs to be shorter, use 9 slides:

1. Title and research question.
2. PDE rollout framing.
3. FNO spatial operator.
4. SSM temporal memory.
5. Attention baseline and fair comparison protocol.
6. Datasets and model coverage.
7. Navier-Stokes2D and SM-FNO2D v2.
8. Cost-efficiency reporting mechanics.
9. Limitations and future work.

## Backup Evidence Paths

- Heat/Burgers aggregate:
  `results/tables/aggregate_metrics.md`
- M9 cost report:
  `results/tables/m9_cost_efficiency_report.md`
- M10 Navier-Stokes2D aggregate:
  `results/tables/navier_stokes2d_aggregate_metrics.md`
- M11 Navier-Stokes2D v2 aggregate:
  `results/tables/navier_stokes2d_m11_aggregate_metrics.md`
- M11 v2 cost report:
  `results/tables/m11_navier_stokes2d_v2_cost_report.md`
