# M8 Burgers1D PDE Extension

## Objective

Extend the existing fair-comparison protocol from Heat1D to a CPU-friendly
viscous Burgers1D dataset and experiment suite.

M8 is protocol-extension work only. Generated metrics are smoke and
protocol-validation sanity outputs, not benchmark results or model rankings.

## Requirements

- Implement a deterministic CPU-friendly viscous Burgers1D data generator.
- Preserve trajectory shape: `[samples, time, space, channels]`.
- Preserve model shape convention:
  `[batch, time, space, channels] -> [batch, pred_time, space, channels]`.
- Add Burgers1D data-generation support to `scripts/generate_data.py`.
- Add Burgers1D smoke configs for:
  - `FNO1D`
  - `SpectralMemoryFNO1D`
  - `Transformer1DBaseline`
- Add Burgers1D rollout-20 protocol-validation configs for the same models.
- Keep configs CPU-friendly and aligned with the Heat1D protocol where
  appropriate.
- Verify train/evaluate/plot for FNO, SM-FNO, and Transformer smoke configs.
- Verify repeated-seed execution for at least FNO and SM-FNO smoke configs.
- Verify metric aggregation includes Burgers1D groups.
- Keep generated artifacts ignored.
- Do not add performance claims.

## Implemented Scope

- Burgers1D generator:
  - `src/sm_fno/data/burgers1d.py`
- Data config:
  - `configs/data/burgers1d.yaml`
- Smoke configs:
  - `configs/experiment/burgers1d_fno_smoke.yaml`
  - `configs/experiment/burgers1d_sm_fno_smoke.yaml`
  - `configs/experiment/burgers1d_transformer_smoke.yaml`
- Rollout-20 protocol-validation configs:
  - `configs/experiment/burgers1d_fno_rollout20.yaml`
  - `configs/experiment/burgers1d_sm_fno_rollout20.yaml`
  - `configs/experiment/burgers1d_transformer_rollout20.yaml`

## Burgers1D Generator

The generator solves the periodic viscous Burgers equation:

```text
u_t + (0.5 * u^2)_x = viscosity * u_xx
```

It uses smooth random low-frequency periodic initial conditions, a conservative
Rusanov flux for nonlinear advection, and a centered finite-difference diffusion
term. The implementation checks advective CFL, diffusion stability, and the
combined explicit convection-diffusion guard

`CFL + 2 * diffusion <= stability_safety`

before the first update and before every time step. Unstable custom
configurations are rejected with a clear `ValueError` before non-finite
trajectories can be written.

This solver is intended for protocol validation and smoke experiments. It is
not a high-accuracy production CFD solver.

## Verification Commands

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/burgers1d.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/burgers1d_fno_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/burgers1d_fno_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/burgers1d_fno_smoke.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/burgers1d_sm_fno_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/burgers1d_sm_fno_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/burgers1d_sm_fno_smoke.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/burgers1d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/burgers1d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/burgers1d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/burgers1d_fno_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/burgers1d_sm_fno_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/*_eval_metrics.json" --output results/tables/aggregate_metrics.json
PYTHONPATH=src python3 scripts/generate_protocol_report.py --aggregate results/tables/aggregate_metrics.json --output results/tables/protocol_summary.md
git diff --check
```

The rollout-20 configs were also run through train/evaluate/plot as
protocol-validation checks.

## Done Criteria

- Tests pass.
- Burgers1D data generation works.
- Burgers FNO, SM-FNO, and Transformer smoke train/evaluate/plot works.
- Burgers repeated-seed execution works for FNO and SM-FNO.
- Aggregation includes Burgers groups separately.
- Burgers rollout-20 configs exist and run.
- Generated Burgers artifacts remain ignored by git.
- M8 decision log and verification report document implementation decisions,
  commands run, artifact paths, and limitations.
