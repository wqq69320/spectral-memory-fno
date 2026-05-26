# M8 Verification Report

Metric values in this report are smoke and protocol-validation sanity checks
only. They are not benchmark claims and must not be used to rank models.

## Code Review Fix

The M8 code review found that the Burgers1D generator checked advective CFL
and diffusion stability separately, which allowed some custom configurations
to violate the combined explicit convection-diffusion limit and produce NaN
trajectories. The generator now checks the current state before each update and
raises `ValueError` when `CFL + 2 * diffusion > stability_safety`, with the
error message reporting the CFL, diffusion number, combined value, and safety
threshold. The default `stability_safety` is `0.95`.

## Commands Run

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q tests/test_burgers1d_data.py` | Pass | 3 Burgers data tests passed, including the unstable combined-stability regression. |
| `PYTHONPATH=src pytest -q` | Pass | 22 tests passed. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/burgers1d.yaml` | Pass | Wrote `data/processed/burgers1d/burgers1d.npz` with shape `(128, 50, 64, 1)`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/burgers1d_fno_smoke.yaml` | Pass | Wrote Burgers FNO smoke checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/burgers1d_fno_smoke.yaml` | Pass | Wrote Burgers FNO smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/burgers1d_fno_smoke.yaml` | Pass | Wrote Burgers FNO smoke figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/burgers1d_sm_fno_smoke.yaml` | Pass | Wrote Burgers SM-FNO smoke checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/burgers1d_sm_fno_smoke.yaml` | Pass | Wrote Burgers SM-FNO smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/burgers1d_sm_fno_smoke.yaml` | Pass | Wrote Burgers SM-FNO smoke figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/burgers1d_transformer_smoke.yaml` | Pass | Wrote Burgers Transformer smoke checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/burgers1d_transformer_smoke.yaml` | Pass | Wrote Burgers Transformer smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/burgers1d_transformer_smoke.yaml` | Pass | Wrote Burgers Transformer smoke figures. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/burgers1d_fno_smoke.yaml --seeds 0 1` | Pass | Wrote Burgers FNO seed 0/1 metrics, checkpoints, and predictions. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/burgers1d_sm_fno_smoke.yaml --seeds 0 1` | Pass | Wrote Burgers SM-FNO seed 0/1 metrics, checkpoints, and predictions. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/burgers1d_fno_rollout20.yaml` | Pass | Protocol-validation config ran successfully. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/burgers1d_fno_rollout20.yaml` | Pass | Wrote Burgers FNO rollout-20 eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/burgers1d_fno_rollout20.yaml` | Pass | Wrote Burgers FNO rollout-20 figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/burgers1d_sm_fno_rollout20.yaml` | Pass | Protocol-validation config ran successfully. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/burgers1d_sm_fno_rollout20.yaml` | Pass | Wrote Burgers SM-FNO rollout-20 eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/burgers1d_sm_fno_rollout20.yaml` | Pass | Wrote Burgers SM-FNO rollout-20 figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/burgers1d_transformer_rollout20.yaml` | Pass | Protocol-validation config ran successfully. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/burgers1d_transformer_rollout20.yaml` | Pass | Wrote Burgers Transformer rollout-20 eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/burgers1d_transformer_rollout20.yaml` | Pass | Wrote Burgers Transformer rollout-20 figures. |
| `PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/*_eval_metrics.json" --output results/tables/aggregate_metrics.json` | Pass | Aggregate outputs include Burgers groups separately. |
| `PYTHONPATH=src python3 scripts/generate_protocol_report.py --aggregate results/tables/aggregate_metrics.json --output results/tables/protocol_summary.md` | Pass | Generated protocol summary includes Burgers groups. |
| `git diff --check` | Pass | No whitespace errors. |
| `git check-ignore -v ...` | Pass | Confirmed generated Burgers data, predictions, checkpoints, figures, and metrics are ignored. |

## Output Artifacts

Generated artifacts remain git-ignored local outputs.

- Dataset:
  - `data/processed/burgers1d/burgers1d.npz`
- Burgers smoke artifacts:
  - `results/checkpoints/burgers1d_fno_smoke.pt`
  - `results/checkpoints/burgers1d_sm_fno_smoke.pt`
  - `results/checkpoints/burgers1d_transformer_smoke.pt`
  - `results/tables/burgers1d_fno_smoke_eval_metrics.json`
  - `results/tables/burgers1d_sm_fno_smoke_eval_metrics.json`
  - `results/tables/burgers1d_transformer_smoke_eval_metrics.json`
  - `outputs/burgers1d_fno_smoke/predictions.npz`
  - `outputs/burgers1d_sm_fno_smoke/predictions.npz`
  - `outputs/burgers1d_transformer_smoke/predictions.npz`
- Burgers repeated-seed artifacts:
  - `results/checkpoints/burgers1d_fno_smoke_seed0.pt`
  - `results/checkpoints/burgers1d_fno_smoke_seed1.pt`
  - `results/checkpoints/burgers1d_sm_fno_smoke_seed0.pt`
  - `results/checkpoints/burgers1d_sm_fno_smoke_seed1.pt`
  - `results/tables/burgers1d_fno_smoke_seed0_eval_metrics.json`
  - `results/tables/burgers1d_fno_smoke_seed1_eval_metrics.json`
  - `results/tables/burgers1d_sm_fno_smoke_seed0_eval_metrics.json`
  - `results/tables/burgers1d_sm_fno_smoke_seed1_eval_metrics.json`
  - `outputs/repeated_seeds/burgers1d_fno_smoke/seed_0/predictions.npz`
  - `outputs/repeated_seeds/burgers1d_fno_smoke/seed_1/predictions.npz`
  - `outputs/repeated_seeds/burgers1d_sm_fno_smoke/seed_0/predictions.npz`
  - `outputs/repeated_seeds/burgers1d_sm_fno_smoke/seed_1/predictions.npz`
- Burgers rollout-20 artifacts:
  - `results/checkpoints/burgers1d_fno_rollout20.pt`
  - `results/checkpoints/burgers1d_sm_fno_rollout20.pt`
  - `results/checkpoints/burgers1d_transformer_rollout20.pt`
  - `results/tables/burgers1d_fno_rollout20_eval_metrics.json`
  - `results/tables/burgers1d_sm_fno_rollout20_eval_metrics.json`
  - `results/tables/burgers1d_transformer_rollout20_eval_metrics.json`
  - `outputs/burgers1d_fno_rollout20/predictions.npz`
  - `outputs/burgers1d_sm_fno_rollout20/predictions.npz`
  - `outputs/burgers1d_transformer_rollout20/predictions.npz`
- Aggregate/report outputs:
  - `results/tables/aggregate_metrics.json`
  - `results/tables/aggregate_metrics.md`
  - `results/tables/protocol_summary.md`

## Burgers Sanity Values

These values verify that Burgers1D executes through the protocol. They are not
performance claims.

| Experiment | Run Type | Count | Seeds | MSE Mean | Relative L2 Mean | Rollout Relative L2 Mean |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `burgers1d_fno_smoke` | `single` | 1 | 42 | 0.0006024881 | 0.0429081550 | 0.0502662659 |
| `burgers1d_sm_fno_smoke` | `single` | 1 | 42 | 0.0006088878 | 0.0433096266 | 0.1161670014 |
| `burgers1d_transformer_smoke` | `single` | 1 | 42 | 0.0000429881 | 0.0115755636 | 0.0312668048 |
| `burgers1d_fno_smoke` | `repeated_seed` | 2 | 0, 1 | 0.0001717033 | 0.0230370155 | 0.0272596218 |
| `burgers1d_sm_fno_smoke` | `repeated_seed` | 2 | 0, 1 | 0.0005835616 | 0.0442561866 | 0.1143676341 |

## Pass/Fail Summary

- Burgers1D data generation works.
- Burgers1D rejects custom configurations that violate the combined explicit
  convection-diffusion stability guard before stepping.
- Burgers smoke train/evaluate/plot works for FNO, SM-FNO, and Transformer.
- Repeated-seed execution works for Burgers FNO and SM-FNO smoke configs.
- Aggregation includes Burgers groups separately.
- Burgers rollout-20 configs exist and run.
- Tests cover Burgers generator shape and smoke config compatibility.
- Generated Burgers artifacts remain ignored by git.
- README, plan, decision log, and verification report are updated without
  performance claims.

## Remaining Risks

- The Burgers solver is a simple explicit finite-difference implementation for
  protocol validation.
- The dataset is small, smooth, and CPU-friendly.
- Two repeated seeds verify mechanics only, not model robustness.
- Rollout-20 configs validate the longer rollout path, not final benchmarks.
- Timing fields are wall-clock CPU measurements from small runs.
