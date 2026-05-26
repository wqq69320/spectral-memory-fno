# M7 Transformer1D Attention Baseline

## Objective

Add a small CPU-friendly `Transformer1DBaseline` and integrate it into the
Heat1D fair-comparison protocol from M6.

M7 is protocol-enabling baseline work only. Smoke, rollout, repeated-seed, and
aggregate outputs are sanity checks for the experiment pipeline, not benchmark
results or model rankings.

## Requirements

- Preserve input shape convention: `[batch, time, space, channels]`.
- Preserve output shape convention: `[batch, pred_time, space, channels]`.
- Keep legacy one-step `[batch, space, channels]` support for existing smoke
  shape checks.
- Add the Transformer model to the config-driven `build_model` path.
- Add a Heat1D Transformer smoke config using the same protocol as MLP, FNO,
  and SM-FNO:
  - `input_steps: 10`
  - `pred_steps: 1`
  - `rollout_steps: 5`
  - `epochs: 2`
  - `batch_size: 64`
  - `learning_rate: 0.001`
  - `weight_decay: 0.0001`
  - `grad_clip_norm: 1.0`
  - `device: cpu`
  - `seed: 42`
  - `train_ratio: 0.8`
  - `val_ratio: 0.1`
- Add a rollout-20 Transformer protocol-validation config.
- Ensure repeated-seed execution works through `scripts/run_repeated_seeds.py`.
- Ensure aggregation includes Transformer metrics as separate groups.
- Keep generated artifacts ignored.
- Do not add performance claims.

## Implemented Scope

- `Transformer1DBaseline` applies temporal self-attention independently at each
  spatial grid point.
- The model maps Heat1D input windows to prediction windows and remains
  CPU-friendly through small smoke config overrides.
- `scripts/train.py` supports `transformer`, `transformer1d`, and
  `transformer1dbaseline` model names.
- New configs:
  - `configs/experiment/heat1d_transformer_smoke.yaml`
  - `configs/experiment/heat1d_transformer_rollout20.yaml`

## Verification Commands

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/heat1d.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_transformer_smoke.yaml
PYTHONPATH=src python3 scripts/run_repeated_seeds.py --config configs/experiment/heat1d_transformer_smoke.yaml --seeds 0 1
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_transformer_rollout20.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_transformer_rollout20.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_transformer_rollout20.yaml
PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/*_eval_metrics.json" --output results/tables/aggregate_metrics.json
PYTHONPATH=src python3 scripts/generate_protocol_report.py --aggregate results/tables/aggregate_metrics.json --output results/tables/protocol_summary.md
git diff --check
```

## Done Criteria

- Tests pass.
- Transformer smoke train/evaluate/plot works.
- Transformer repeated-seed execution works.
- Aggregation includes Transformer groups separately.
- Transformer rollout-20 protocol-validation config runs.
- Generated Transformer artifacts remain ignored by git.
- M7 decision log and verification report document implementation decisions,
  commands run, artifact paths, and limitations.
