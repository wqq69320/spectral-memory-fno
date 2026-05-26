# M2-M5 Decision Log

## Decisions Made

- Use the repository-wide forecasting convention
  `[batch, time, space, channels] -> [batch, pred_time, space, channels]` for
  Heat1D train/evaluate paths.
- Keep legacy one-step `[batch, space, channels]` support in `FNO1D`,
  `SpectralMemoryFNO1D`, and rollout helpers for existing smoke scripts and
  lightweight shape checks.
- Implement `SpectralConv1D` with `torch.fft.rfft`/`irfft` and learned complex
  weights on retained Fourier modes.
- Implement `DiagonalSSM` as a clear recurrent block with
  `h_t = A h_{t-1} + B x_t`, using a learned diagonal transition constrained
  with `tanh` for small smoke-run stability.
- Implement `SpectralMemoryFNO1D` by applying FNO spatial blocks independently
  to each input state, then applying the diagonal SSM across the input window for
  each spatial position.
- Add experiment-local `model_overrides` so smoke configs can stay CPU-friendly
  without creating separate model definition files.
- Record inference timing and rollout metrics in evaluation JSON files rather
  than only printing them to stdout.
- Configure plotting to use a noninteractive Matplotlib backend and writable
  temp cache directories so verification commands run in sandboxed environments.

## Alternatives Considered

- A full Transformer baseline was deferred because this milestone focuses on
  the FNO and SM-FNO Heat1D MVP and the plan explicitly says not to implement
  the full Transformer unless the required work is complete and stable.
- A standalone trainable SSM baseline was not added. The requirement can be
  satisfied by a documented `DiagonalSSM` temporal block used by SM-FNO, and a
  separate baseline would add another experiment axis before the core pipeline
  is validated.
- A Mamba-style selective scan was deferred because it would add optimization
  complexity that is not needed for CPU smoke verification.
- Multi-step direct decoding is supported through `pred_steps`, but the smoke
  configs use one-step decoding with autoregressive rollout to keep the
  comparisons simple and CPU-friendly.

## Implementation Rationale

- Flattening the input time window into the FNO lift for `FNO1D` keeps the
  baseline spatially focused while preserving the shared input/output shape
  convention.
- Running `DiagonalSSM` over per-grid latent sequences in `SpectralMemoryFNO1D`
  makes temporal memory explicit without using attention.
- Small smoke overrides (`modes: 8`, `width: 16`, `depth: 2`) keep verification
  commands fast enough to run routinely on CPU.
- Rollout metrics are saved as sanity checks for pipeline behavior only. They
  are not benchmark evidence.

## Known Limitations

- Smoke metrics are sanity checks only and should not be interpreted as model
  rankings.
- Heat1D smoke data is small and smooth; it is not enough for research claims.
- Rollout evaluation currently uses the held-out Heat1D trajectories from the
  same generated dataset and a short rollout horizon.
- The diagonal SSM is intentionally simple and not a structured selective-scan
  implementation.
- Transformer baseline training remains future work.
- Inference timing uses Python wall-clock timing around small CPU batches and is
  useful only for confirming that timing is logged.

## Follow-Up Work

- Add a standalone SSM baseline config if future comparisons require it.
- Add longer rollout horizons and repeated seeds after the MVP pipeline is
  stable.
- Add richer experiment reports that aggregate metrics across seeds.
- Extend the data/model path to Burgers1D after Heat1D verification remains
  stable.
- Consider profiling and optimized SSM implementations only after correctness
  and fair-comparison protocols are established.
