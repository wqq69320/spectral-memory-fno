# M2-M5 Research MVP

## Goal

Complete a working Heat1D research MVP that supports config-driven training,
evaluation, plotting, and basic rollout comparison for:

1. `MLPBaseline`
2. `FNO1D` baseline
3. `DiagonalSSM` temporal memory block
4. `SpectralMemoryFNO1D` proposed model

This milestone is pipeline-enabling work only. It does not establish that one
model outperforms another.

## Requirements

- Preserve input shape convention: `[batch, time, space, channels]`.
- Preserve output shape convention: `[batch, pred_time, space, channels]`.
- Implement a real `FNO1D` baseline using `torch.fft.rfft`/`irfft` and learnable
  complex spectral weights.
- Implement `DiagonalSSM` with the documented recurrence
  `h_t = A h_{t-1} + B x_t`.
- Implement `SpectralMemoryFNO1D` by combining FNO spatial mixing with SSM
  temporal memory.
- Keep Heat1D smoke experiments CPU-friendly.
- Add Heat1D FNO and Heat1D SM-FNO smoke experiment configs.
- Add basic autoregressive rollout evaluation.
- Add per-timestep rollout relative L2 error.
- Add basic inference-time logging.
- Keep experiment execution config-driven with plain YAML.
- Do not implement a full Transformer baseline in this milestone.
- Do not add benchmark claims.
- Do not over-optimize CUDA kernels or Mamba-style selective scan.

## Implemented Scope

- `SpectralConv1D` uses retained Fourier modes with learnable complex weights.
- `FNO1D` maps Heat1D input windows to prediction windows.
- `DiagonalSSM` provides a small diagonal recurrent temporal block.
- `SpectralMemoryFNO1D` applies FNO blocks to each input state and applies the
  diagonal SSM over the input window for each spatial position.
- Training/evaluation scripts support `mlp`, `fno1d`, and `sm_fno1d` model
  configs, including experiment-local `model_overrides`.
- Evaluation writes one-step metrics, rollout metrics, per-timestep rollout
  relative L2, and inference timing fields.
- Plotting writes training-loss, sample-prediction, and rollout-error figures.

## Smoke Configs

- `configs/experiment/heat1d_mlp_smoke.yaml`
- `configs/experiment/heat1d_fno_smoke.yaml`
- `configs/experiment/heat1d_sm_fno_smoke.yaml`

All three use the same Heat1D smoke protocol: `input_steps: 10`,
`pred_steps: 1`, `rollout_steps: 5`, `epochs: 2`, `batch_size: 64`, CPU
execution, seed `42`, and the same train/validation split. FNO and SM-FNO use
small model overrides to keep verification CPU-friendly.

## Verification Commands

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/heat1d.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_mlp_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_mlp_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_mlp_smoke.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_fno_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_fno_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_fno_smoke.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/heat1d_sm_fno_smoke.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/heat1d_sm_fno_smoke.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/heat1d_sm_fno_smoke.yaml
```

## Done Criteria

- All tests pass.
- Heat1D FNO smoke train/evaluate/plot works.
- Heat1D SM-FNO smoke train/evaluate/plot works.
- Basic rollout evaluation works.
- Verification report lists commands run, pass/fail status, generated artifact
  paths, and limitations.
- README current status is updated honestly without performance claims.
- No fake results or unsupported claims are added.
