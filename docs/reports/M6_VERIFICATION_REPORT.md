# M6 Verification Report

Metric values in this report are smoke and protocol-validation sanity checks
only. They are not benchmark claims and must not be used to rank models.

## Commands Run

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q` | Pass | 17 tests passed. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/heat1d.yaml` | Pass | Wrote `data/processed/heat1d/heat1d.npz` with shape `(128, 50, 64, 1)`. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/heat1d_fno_smoke.yaml --seeds 0 1` | Pass | Wrote FNO seed 0/1 metrics, checkpoints, and predictions. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/heat1d_sm_fno_smoke.yaml --seeds 0 1` | Pass | Wrote SM-FNO seed 0/1 metrics, checkpoints, and predictions. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_mlp_smoke.yaml` | Pass | Refreshed smoke metrics with M6 metadata. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_mlp_smoke.yaml` | Pass | Wrote MLP smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_mlp_smoke.yaml` | Pass | Wrote MLP smoke figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_fno_smoke.yaml` | Pass | Refreshed smoke metrics with M6 metadata. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_fno_smoke.yaml` | Pass | Wrote FNO smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_fno_smoke.yaml` | Pass | Wrote FNO smoke figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_sm_fno_smoke.yaml` | Pass | Refreshed smoke metrics with M6 metadata. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_sm_fno_smoke.yaml` | Pass | Wrote SM-FNO smoke figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_mlp_rollout20.yaml` | Pass | Protocol-validation config ran successfully. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_mlp_rollout20.yaml` | Pass | Wrote rollout-20 MLP eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_mlp_rollout20.yaml` | Pass | Wrote rollout-20 MLP figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_fno_rollout20.yaml` | Pass | Protocol-validation config ran successfully. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_fno_rollout20.yaml` | Pass | Wrote rollout-20 FNO eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_fno_rollout20.yaml` | Pass | Wrote rollout-20 FNO figures. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_sm_fno_rollout20.yaml` | Pass | Protocol-validation config ran successfully. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_sm_fno_rollout20.yaml` | Pass | Wrote rollout-20 SM-FNO eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_sm_fno_rollout20.yaml` | Pass | Wrote rollout-20 SM-FNO figures. |
| `PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/*_eval_metrics.json" --output results/tables/aggregate_metrics.json` | Pass | Wrote aggregate JSON and Markdown. |
| `PYTHONPATH=src python3 scripts/generate_protocol_report.py --aggregate results/tables/aggregate_metrics.json --output results/tables/protocol_summary.md` | Pass | Wrote generated protocol summary. |
| `git diff --check` | Pass | No whitespace errors. |

## Output Artifacts

Generated artifacts remain git-ignored local outputs.

- Dataset:
  - `data/processed/heat1d/heat1d.npz`
- Repeated-seed FNO artifacts:
  - `results/checkpoints/heat1d_fno_smoke_seed0.pt`
  - `results/checkpoints/heat1d_fno_smoke_seed1.pt`
  - `results/tables/heat1d_fno_smoke_seed0_train_metrics.json`
  - `results/tables/heat1d_fno_smoke_seed1_train_metrics.json`
  - `results/tables/heat1d_fno_smoke_seed0_eval_metrics.json`
  - `results/tables/heat1d_fno_smoke_seed1_eval_metrics.json`
  - `outputs/repeated_seeds/heat1d_fno_smoke/seed_0/predictions.npz`
  - `outputs/repeated_seeds/heat1d_fno_smoke/seed_1/predictions.npz`
- Repeated-seed SM-FNO artifacts:
  - `results/checkpoints/heat1d_sm_fno_smoke_seed0.pt`
  - `results/checkpoints/heat1d_sm_fno_smoke_seed1.pt`
  - `results/tables/heat1d_sm_fno_smoke_seed0_train_metrics.json`
  - `results/tables/heat1d_sm_fno_smoke_seed1_train_metrics.json`
  - `results/tables/heat1d_sm_fno_smoke_seed0_eval_metrics.json`
  - `results/tables/heat1d_sm_fno_smoke_seed1_eval_metrics.json`
  - `outputs/repeated_seeds/heat1d_sm_fno_smoke/seed_0/predictions.npz`
  - `outputs/repeated_seeds/heat1d_sm_fno_smoke/seed_1/predictions.npz`
- Rollout-20 protocol-validation artifacts:
  - `results/tables/heat1d_mlp_rollout20_eval_metrics.json`
  - `results/tables/heat1d_fno_rollout20_eval_metrics.json`
  - `results/tables/heat1d_sm_fno_rollout20_eval_metrics.json`
  - `outputs/heat1d_mlp_rollout20/predictions.npz`
  - `outputs/heat1d_fno_rollout20/predictions.npz`
  - `outputs/heat1d_sm_fno_rollout20/predictions.npz`
  - `results/figures/heat1d_mlp_rollout20_rollout_relative_l2.png`
  - `results/figures/heat1d_fno_rollout20_rollout_relative_l2.png`
  - `results/figures/heat1d_sm_fno_rollout20_rollout_relative_l2.png`
- Aggregate/report artifacts:
  - `results/tables/aggregate_metrics.json`
  - `results/tables/aggregate_metrics.md`
  - `results/tables/protocol_summary.md`

## Aggregate Sanity Values

The aggregate table in `results/tables/aggregate_metrics.md` includes single
smoke runs, two-seed repeated runs for FNO and SM-FNO, and rollout-20
protocol-validation runs. The repeated-seed groups were:

| Experiment | Run Type | Count | Seeds | MSE Mean | Relative L2 Mean | Rollout Relative L2 Mean |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `heat1d_fno_smoke` | `repeated_seed` | 2 | 0, 1 | 0.0002729532 | 0.0249905307 | 0.0283573829 |
| `heat1d_sm_fno_smoke` | `repeated_seed` | 2 | 0, 1 | 0.0006092322 | 0.0420814973 | 0.1105882265 |

These values are recorded only to verify aggregation behavior.

## Pass/Fail Summary

- Shared Heat1D comparison protocol is documented in
  `docs/plans/M6_FAIR_COMPARISON_PROTOCOL.md`.
- Repeated-seed execution works for FNO and SM-FNO smoke configs.
- Aggregation writes `aggregate_metrics.json` and `aggregate_metrics.md`.
- Protocol summary generation writes `protocol_summary.md`.
- Rollout-20 configs exist and run for MLP, FNO, and SM-FNO.
- Tests cover toy aggregation and repeated-seed config expansion.
- Generated artifacts remain ignored by git.
- No Burgers1D or Transformer training implementation was added.
- No benchmark claims were added.

## Remaining Risks

- Verification uses two repeated seeds to keep CPU runtime small.
- The generated Heat1D dataset is small and smooth.
- Rollout-20 configs validate the protocol path only.
- Timing metrics are wall-clock CPU measurements from small runs.
- Aggregation currently summarizes scalar JSON metrics only and does not yet
  produce statistical significance tests.
