# M9 Decision Log

## Decisions Made

- Add lightweight parameter-count helpers under `sm_fno.evaluation.costs` and
  log total, trainable, and frozen parameter counts from both train and
  evaluate scripts.
- Keep timing measurements as wall-clock CPU timings from the existing
  evaluation path, then normalize rollout timing into seconds per step and
  seconds per example per step.
- Use a config-driven horizon sweep in `scripts/run_horizon_sweep.py` rather
  than adding many static per-horizon experiment YAML files.
- Train each base experiment once per seed, then reuse that checkpoint across
  horizon-specific evaluations.
- Use the existing small Heat1D and Burgers1D smoke configs for FNO1D, SM-FNO,
  and Transformer1D with horizons `5`, `10`, and `20` and seeds `0`, `1`.
- Add a cost-specific aggregator and plotter instead of changing the M6
  aggregate table semantics.
- Write generated M9 metrics, expanded configs, predictions, checkpoints,
  plots, and generated reports under existing ignored `results/` and
  `outputs/` paths.

## Alternatives Considered

- Adding separate `*_rollout5.yaml`, `*_rollout10.yaml`, and `*_rollout20.yaml`
  configs for every dataset/model pair was rejected because a runner keeps the
  sweep less repetitive and makes seed expansion explicit.
- Measuring FLOPs or peak memory was deferred because reliable cross-platform
  instrumentation would add complexity beyond M9's CPU-friendly protocol scope.
- Reusing only single-seed smoke outputs was rejected because the analysis layer
  should exercise repeated-seed aggregation where practical.
- Training a separate checkpoint for each horizon was rejected because horizon
  only changes evaluation rollout length for the current one-step configs.

## Rationale

- Parameter counts and normalized rollout timing are simple, repeatable cost
  fields that can be recorded for every config-driven run.
- Reusing checkpoints across horizons isolates the horizon variable during M9
  analysis mechanics.
- Keeping M9 aggregation separate from M6 aggregation avoids changing the
  earlier fair-comparison report format while still supporting cost-specific
  summaries.
- Small CPU runs are sufficient to validate analysis mechanics without implying
  final model quality or efficiency.

## Known Limitations

- M9 timing values are local wall-clock CPU measurements and may vary by
  machine load, PyTorch version, and thread scheduling.
- The parameter-count metric does not capture FLOPs, activation memory, cache
  effects, or rollout implementation overhead in detail.
- Heat1D and Burgers1D smoke datasets are small protocol fixtures.
- Two seeds exercise repeated-seed reporting mechanics only.
- M9 outputs are protocol-scale analysis artifacts and must not be interpreted
  as final benchmark claims or model rankings.

## Follow-Up Work

- Add hardware and thread metadata to timing records before larger comparisons.
- Add optional memory profiling and operation-count estimates if they can be
  collected consistently.
- Run larger seed sets and dataset sizes only after the protocol-scale tooling
  is stable.
- Add final benchmark reports only when configs, metrics, plots, and validation
  commands support the claims being made.
