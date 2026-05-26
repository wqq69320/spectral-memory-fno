# M6 Decision Log

## Decisions Made

- Keep M6 focused on Heat1D protocol hardening rather than adding new PDE tasks.
- Add `scripts/run_repeated_seeds.py` as a small wrapper around the existing
  config-driven `train.py` and `evaluate.py` scripts.
- Expand repeated-seed configs in memory and write temporary YAML files for
  subprocess execution, while writing durable outputs to clearly named local
  artifact paths.
- Store repeated-seed metrics in `results/tables/*_seed*_eval_metrics.json` so
  they are easy to aggregate with a simple glob.
- Add protocol metadata to train/evaluate metrics: `base_experiment`,
  `run_type`, `seed`, `input_steps`, `pred_steps`, and `rollout_steps`.
- Aggregate metrics by `base_experiment` and `run_type` so single smoke runs and
  repeated-seed runs are summarized separately.
- Write both JSON and Markdown aggregate outputs in `results/tables/` so the
  generated summaries remain ignored by git.
- Add rollout-20 configs as protocol-validation configs only.

## Alternatives Considered

- A larger experiment-grid runner was deferred. M6 only needs repeated seeds for
  one base config at a time.
- Persisting expanded per-seed YAML configs was deferred. The generated metrics
  and checkpoints are already seed-named, and the base config plus seed list is
  enough to reproduce the run.
- Burgers1D and Transformer training were deferred because M6 is about protocol
  hardening for the existing Heat1D MVP.
- Aggregating every possible metric was deferred. M6 focuses on MSE, relative
  L2, rollout relative L2, and inference timing fields already produced by
  evaluation.

## Rationale

- Running the existing scripts through a repeated-seed wrapper preserves the
  repository's YAML-driven execution model.
- Keeping repeated-seed outputs in ignored local artifact directories avoids
  committing generated datasets, checkpoints, predictions, or metric tables.
- Separating `single` and `repeated_seed` groups prevents smoke metrics from
  being merged into repeated-seed summaries.
- Rollout-20 configs exercise a longer-horizon protocol path without implying a
  final benchmark setting.

## Known Limitations

- M6 repeated-seed verification uses only two seeds to keep commands
  CPU-friendly.
- Aggregated values are sanity checks for protocol mechanics, not performance
  claims.
- Inference timing is wall-clock timing from small CPU runs.
- The Transformer remains a placeholder and is not part of M6 execution.
- Burgers1D is intentionally not implemented for M6.

## Follow-Up Work

- Add larger seed sets once the protocol is stable.
- Add confidence intervals or nonparametric summaries for final reports.
- Add Transformer and Burgers1D only after Heat1D protocol validation remains
  stable.
- Add richer provenance capture if experiments move beyond smoke-scale runs.
