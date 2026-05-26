# M7 Decision Log

## Decisions Made

- Implement `Transformer1DBaseline` as a temporal-attention baseline that runs
  self-attention over the input time window independently for each grid point.
- Preserve legacy one-step `[batch, space, channels]` support so existing smoke
  tests and simple callers keep working.
- Add `input_steps` and `pred_steps` to the Transformer constructor to match the
  repository-wide forecast shape convention.
- Use small Transformer smoke overrides (`d_model: 32`, `num_layers: 1`,
  `dim_feedforward: 64`) to keep Heat1D verification CPU-friendly.
- Integrate Transformer through the existing YAML-driven `train.py`,
  `evaluate.py`, `plot_results.py`, repeated-seed runner, and metric aggregator.
- Add Transformer rollout-20 as a protocol-validation config, not a final
  benchmark config.

## Alternatives Considered

- Flattening time and space into one attention sequence was deferred because
  the smoke input would produce much longer token sequences and would be less
  CPU-friendly.
- Adding spatial attention plus temporal attention was deferred to avoid
  increasing the baseline complexity before protocol integration is verified.
- Adding Burgers1D or other tasks was deferred because M7 is scoped to Heat1D
  fair-comparison integration.

## Rationale

- Temporal attention per grid point gives a clear attention-based baseline for
  sequence modeling while keeping the implementation small and reproducible.
- Reusing the existing train/evaluate/plot scripts keeps Transformer runs under
  the same M6 protocol machinery as MLP, FNO, and SM-FNO.
- Separate artifact paths prevent Transformer outputs from overwriting existing
  model outputs.

## Known Limitations

- The Transformer baseline does not mix spatial positions directly; it models
  temporal history independently per grid point.
- Smoke and repeated-seed runs are small CPU protocol checks only.
- The rollout-20 config validates the longer rollout path but is not a final
  benchmark.
- Aggregated values are sanity outputs and should not be used as model
  rankings.

## Follow-Up Work

- Consider spatial-temporal attention variants after the fair-comparison
  protocol remains stable.
- Add larger seed sets only after CPU smoke checks stay reliable.
- Extend Transformer comparisons beyond Heat1D in a later milestone.
