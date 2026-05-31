# M14 Rollout-Stable SM-FNO2D Plan

## Objective

Fix the medium Navier-Stokes2D autoregressive rollout collapse observed in
SM-FNO2D v2 while preserving all existing FNO2D, SM-FNO2D v1/v2, and
Transformer2D behavior. The medium diagnostic protocol uses a 24x24 grid, 76
time steps, 10 input steps, one-step prediction, and a 36-step rollout horizon.

M14 outputs are diagnostic artifacts only. They do not support benchmark
claims, cost-efficiency claims, or model rankings.

## Scope

- Diagnose the one-step versus rollout mismatch with per-timestep rollout
  relative L2, prediction statistics, target statistics, and spatial Fourier
  spectrum error.
- Implement `SpectralMemoryFNO2DV3` as a separate model.
- Use FNO2D as the residual base predictor in v3.
- Use the stable gated SSM path only as a conservative temporal correction:
  `prediction = FNO prediction + gate * SSM correction`.
- Add optional rollout-aware training with `rollout_train_steps` and
  `rollout_loss_weight` config fields.
- Add `configs/model/sm_fno2d_v3.yaml`.
- Add
  `configs/experiment/navier_stokes2d_medium_sm_fno_v3_diagnostic.yaml`.
- Compare v3 against at least FNO2D and SM-FNO2D v2 under the medium
  Navier-Stokes2D protocol.
- Generate train/evaluate/plot outputs and 3D-style visualization for the best
  stable M14 artifact.
- Document whether v3 improves rollout stability over v2.

## Done Criteria

- [x] `SpectralMemoryFNO2DV3` exists and is registered without changing v1/v2,
  FNO2D, or Transformer2D behavior.
- [x] v3 forward/backward shape tests pass.
- [x] Optional rollout-aware training is available through
  `rollout_train_steps` and `rollout_loss_weight`.
- [x] Medium v3 diagnostic config exists.
- [x] Medium FNO2D, SM-FNO2D v2, and SM-FNO2D v3 train/evaluate/plot paths run.
- [x] 36-step rollout metrics are produced for all three compared models.
- [x] Diagnostics include per-timestep rollout relative L2, prediction stats,
  target stats, and rollout spatial spectrum error.
- [x] A 3D-style visualization is generated for the stable v3 artifact.
- [x] `PYTHONPATH=src pytest -q` passes.
- [x] Representative generated artifacts are confirmed git-ignored.
- [x] `docs/reports/M14_VERIFICATION_REPORT.md` records commands, artifacts,
  metrics, limitations, and whether rollout stability improved.

## Required Verification

```bash
PYTHONPATH=src pytest -q tests/test_shapes.py tests/test_imports.py tests/test_navier_stokes2d_data.py tests/test_metrics.py
PYTHONPATH=src python3 scripts/generate_data.py --config configs/data/navier_stokes2d_medium.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml
PYTHONPATH=src python3 scripts/train.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v3_diagnostic.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v3_diagnostic.yaml
PYTHONPATH=src python3 scripts/plot_results.py --config configs/experiment/navier_stokes2d_medium_sm_fno_v3_diagnostic.yaml
PYTHONPATH=src python3 scripts/aggregate_metrics.py --input-glob "results/tables/navier_stokes2d_medium_*fno*_diagnostic_eval_metrics.json" --output results/tables/m14_navier_stokes2d_medium_rollout_stability_metrics.json
PYTHONPATH=src python3 scripts/visualize_navier_stokes2d_3d.py --prediction-path outputs/navier_stokes2d_medium_sm_fno_v3_diagnostic/predictions.npz --output-dir results/figures/m14_navier_stokes2d_v3_3d --prefix navier_stokes2d_medium_sm_fno_v3 --surface-step 17 --max-animation-frames 18 --fps 6 --dpi 120 --velocity-traces
PYTHONPATH=src pytest -q
git diff --check
```

## Non-Goals

- Do not tune final hyperparameters.
- Do not remove or reinterpret SM-FNO2D v1/v2.
- Do not make benchmark or model-ranking claims from M14 diagnostics.
- Do not treat the 24x24 fixture as validated CFD evidence.
