# M9 Cost-Efficiency Verification Report

M9 outputs are protocol-scale analysis artifacts only. They are not final
benchmark claims, they do not rank models, and they should not be used to claim
that one model is more accurate or cost-efficient than another.

## Verification Status

Verification passed on the current code and generated artifacts remain under
ignored local output paths.

## Commands

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q tests/test_m9_cost_efficiency.py` | Pass | 4 focused M9 tests passed. |
| `PYTHONPATH=src pytest -q` | Pass | 26 tests passed. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/heat1d.yaml` | Pass | Wrote `data/processed/heat1d/heat1d.npz` with shape `(128, 50, 64, 1)`. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/burgers1d.yaml` | Pass | Wrote `data/processed/burgers1d/burgers1d.npz` with shape `(128, 50, 64, 1)`. |
| `PYTHONPATH=src python3 scripts/run_horizon_sweep.py --config configs/analysis/m9_cost_efficiency_smoke.yaml` | Pass | Ran Heat1D/Burgers1D FNO1D, SM-FNO, and Transformer1D for horizons 5/10/20 and seeds 0/1; wrote 48 expanded configs. |
| `PYTHONPATH=src python3 scripts/aggregate_cost_metrics.py --input-glob "results/tables/m9_cost_efficiency/*_eval_metrics.json" --output results/tables/m9_cost_efficiency_aggregate.json` | Pass | Aggregated 36 evaluation metric files into 18 dataset/model/horizon groups. |
| `PYTHONPATH=src python3 scripts/plot_cost_efficiency.py --aggregate results/tables/m9_cost_efficiency_aggregate.json --output-dir results/figures/m9_cost_efficiency` | Pass | Wrote 3 cost-efficiency plots. |
| `PYTHONPATH=src python3 scripts/generate_cost_efficiency_report.py --aggregate results/tables/m9_cost_efficiency_aggregate.json --output results/tables/m9_cost_efficiency_report.md` | Pass | Wrote generated protocol-scale report. |
| `git diff --check` | Pass | No whitespace errors. |
| `git check-ignore -v ...` | Pass | Confirmed representative generated M9 data, configs, predictions, checkpoints, metrics, reports, and figures are ignored. |

## Output Artifacts

Generated artifacts:

- `outputs/m9_cost_efficiency_expanded_configs/*.yaml` (48 expanded configs)
- `outputs/m9_cost_efficiency/*/seed_*/horizon_*/predictions.npz`
- `results/checkpoints/*_m9_seed*.pt` (12 checkpoints)
- `results/tables/m9_cost_efficiency/*_train_metrics.json` (12 files)
- `results/tables/m9_cost_efficiency/*_eval_metrics.json` (36 files)
- `results/tables/m9_cost_efficiency_aggregate.json`
- `results/tables/m9_cost_efficiency_aggregate.md`
- `results/tables/m9_cost_efficiency_report.md`
- `results/figures/m9_cost_efficiency/*.png` (3 plots)

These paths were confirmed as local ignored artifacts.

## Aggregate Coverage

The M9 aggregate covers:

- Datasets: `heat1d`, `burgers1d`.
- Models: `fno1d`, `sm_fno1d`, `transformer1d`.
- Horizons: `5`, `10`, `20`.
- Seeds per dataset/model/horizon group: `0`, `1`.

The generated aggregate table reports the recorded values as protocol-scale
sanity outputs only.

## Metrics Recorded

- One-step MSE and relative L2.
- Rollout MSE, rollout relative L2, and per-timestep rollout relative L2.
- Total, trainable, and frozen parameter counts.
- One-step inference seconds, seconds per forward pass, and seconds per example.
- Rollout inference seconds, seconds per rollout step, seconds per example, and
  seconds per example per step.

## Remaining Risks

- Timing fields are local wall-clock CPU measurements.
- Parameter counts are a coarse cost proxy and do not measure FLOPs or memory.
- The datasets and models are intentionally small protocol fixtures.
- Repeated seeds `0` and `1` validate reporting mechanics only.
