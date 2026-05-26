# M7 Verification Report

Metric values in this report are smoke and protocol-validation sanity checks
only. They are not benchmark claims and must not be used to rank models.

## Commands Run

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q` | Pass | 19 tests passed. |
| `PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/heat1d.yaml` | Pass | Wrote `data/processed/heat1d/heat1d.npz` with shape `(128, 50, 64, 1)`. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_transformer_smoke.yaml` | Pass | Wrote Transformer smoke checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_transformer_smoke.yaml` | Pass | Wrote Transformer smoke eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_transformer_smoke.yaml` | Pass | Wrote Transformer smoke training, prediction, and rollout figures. |
| `PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/heat1d_transformer_smoke.yaml --seeds 0 1` | Pass | Wrote Transformer seed 0/1 metrics, checkpoints, and predictions. |
| `PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_transformer_rollout20.yaml` | Pass | Wrote Transformer rollout-20 checkpoint and train metrics. |
| `PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_transformer_rollout20.yaml` | Pass | Wrote Transformer rollout-20 eval metrics and predictions. |
| `PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_transformer_rollout20.yaml` | Pass | Wrote Transformer rollout-20 figures. |
| `PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/*_eval_metrics.json" --output results/tables/aggregate_metrics.json` | Pass | Aggregate outputs include Transformer groups separately. |
| `PYTHONPATH=src python3 scripts/generate_protocol_report.py --aggregate results/tables/aggregate_metrics.json --output results/tables/protocol_summary.md` | Pass | Generated protocol summary includes Transformer groups and no ranking language. |
| `git diff --check` | Pass | No whitespace errors. |

## Output Artifacts

Generated artifacts remain git-ignored local outputs.

- Transformer smoke:
  - `results/checkpoints/heat1d_transformer_smoke.pt`
  - `results/tables/heat1d_transformer_smoke_train_metrics.json`
  - `results/tables/heat1d_transformer_smoke_eval_metrics.json`
  - `outputs/heat1d_transformer_smoke/predictions.npz`
  - `results/figures/heat1d_transformer_smoke_training_loss.png`
  - `results/figures/heat1d_transformer_smoke_prediction.png`
  - `results/figures/heat1d_transformer_smoke_rollout_relative_l2.png`
- Transformer repeated seeds:
  - `results/checkpoints/heat1d_transformer_smoke_seed0.pt`
  - `results/checkpoints/heat1d_transformer_smoke_seed1.pt`
  - `results/tables/heat1d_transformer_smoke_seed0_train_metrics.json`
  - `results/tables/heat1d_transformer_smoke_seed1_train_metrics.json`
  - `results/tables/heat1d_transformer_smoke_seed0_eval_metrics.json`
  - `results/tables/heat1d_transformer_smoke_seed1_eval_metrics.json`
  - `outputs/repeated_seeds/heat1d_transformer_smoke/seed_0/predictions.npz`
  - `outputs/repeated_seeds/heat1d_transformer_smoke/seed_1/predictions.npz`
- Transformer rollout-20 protocol validation:
  - `results/checkpoints/heat1d_transformer_rollout20.pt`
  - `results/tables/heat1d_transformer_rollout20_train_metrics.json`
  - `results/tables/heat1d_transformer_rollout20_eval_metrics.json`
  - `outputs/heat1d_transformer_rollout20/predictions.npz`
  - `results/figures/heat1d_transformer_rollout20_training_loss.png`
  - `results/figures/heat1d_transformer_rollout20_prediction.png`
  - `results/figures/heat1d_transformer_rollout20_rollout_relative_l2.png`
- Aggregate/report outputs:
  - `results/tables/aggregate_metrics.json`
  - `results/tables/aggregate_metrics.md`
  - `results/tables/protocol_summary.md`

## Transformer Sanity Values

These values verify that the Transformer baseline executes through the protocol.
They are not performance claims.

| Experiment | Run Type | Count | Seeds | MSE | Relative L2 | Rollout Relative L2 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `heat1d_transformer_smoke` | `single` | 1 | 42 | 0.0000343899 | 0.0098688532 | 0.0249878373 |
| `heat1d_transformer_smoke` | `repeated_seed` | 2 | 0, 1 | 0.0003876896 | 0.0335746182 | 0.0545668956 |
| `heat1d_transformer_rollout20` | `single` | 1 | 42 | 0.0000343899 | 0.0098688532 | 0.0584166460 |

## Pass/Fail Summary

- `Transformer1DBaseline` follows the repository shape convention.
- Transformer is integrated into the config-driven model builder.
- Transformer smoke train/evaluate/plot works.
- Transformer repeated-seed execution works.
- Aggregation includes Transformer groups separately from MLP, FNO, and SM-FNO.
- Transformer rollout-20 protocol-validation config runs.
- Generated Transformer artifacts remain ignored by git.
- README, plan, decision log, and verification report are updated without
  performance claims.

## Remaining Risks

- The Transformer baseline models temporal attention independently at each grid
  point and does not use spatial attention.
- Smoke and repeated-seed runs use small CPU-friendly settings.
- Two repeated seeds are enough to verify protocol mechanics, not robustness.
- Timing fields are wall-clock CPU measurements from small runs.
- Rollout-20 remains protocol validation, not a final long-horizon benchmark.
