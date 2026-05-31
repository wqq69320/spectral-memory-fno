# M13 3D-Style Navier-Stokes2D Visualization Report

M13 adds presentation-ready visualization tooling for existing Navier-Stokes2D
prediction artifacts. The figures render 2D vorticity fields as 3D-style
surfaces for visual communication. They do not show true 3D Navier-Stokes
flow, and they should not be described as 3D flow forecasting.

## What Is Visualized

- Scalar 2D vorticity fields from existing `predictions.npz` artifacts.
- True, predicted, and absolute-error vorticity panels.
- A rollout animation over time using 3D-style surface panels.
- Optional 2D incompressible velocity traces derived from scalar vorticity by a
  periodic streamfunction solve.

The velocity traces are a qualitative visualization aid only. They are derived
from each selected vorticity snapshot as a stationary 2D field and are not
validated Lagrangian particle trajectories through a time-dependent flow.

## What Is Not Claimed

- No true 3D Navier-Stokes data is used.
- No 3D model or 3D flow forecast is claimed.
- No benchmark claim or model ranking is made.
- No physical validation claim is made for the derived velocity traces.

## Implementation

Added:

- `src/sm_fno/visualization/navier_stokes2d_3d.py`
- `scripts/visualize_navier_stokes2d_3d.py`
- `tests/test_navier_stokes2d_visualization.py`

The script reads an existing `predictions.npz` file and uses rollout arrays
when available:

- `rollout_targets`
- `rollout_predictions`

If rollout arrays are absent, it falls back to one-step `targets` and
`predictions`.

## Commands

| Command | Status | Notes |
| --- | --- | --- |
| `PYTHONPATH=src pytest -q tests/test_navier_stokes2d_visualization.py` | Pass | 4 visualization helper tests passed. |
| `PYTHONPATH=src python3 scripts/visualize_navier_stokes2d_3d.py --prediction-path outputs/navier_stokes2d_medium_sm_fno_v2_diagnostic/predictions.npz --output-dir results/figures/m13_navier_stokes2d_3d --prefix navier_stokes2d_medium_sm_fno_v2 --surface-step 17 --max-animation-frames 18 --fps 6 --dpi 120 --velocity-traces` | Pass | Wrote static surface, rollout GIF, and vorticity-derived velocity trace figure. |
| `python3 - <<'PY' ... PIL image inspection ... PY` | Pass | Confirmed generated image/video dimensions and GIF frame count. |
| `git check-ignore -v ...` | Pass | Confirmed generated M13 visualization artifacts are ignored. |
| `PYTHONPATH=src pytest -q` | Pass | Full test suite passed after adding visualization tooling. |
| `git diff --check` | Pass | No whitespace errors. |

## Generated Artifacts

Generated local presentation artifacts:

- `results/figures/m13_navier_stokes2d_3d/navier_stokes2d_medium_sm_fno_v2_rollout_surface_step18.png`
- `results/figures/m13_navier_stokes2d_3d/navier_stokes2d_medium_sm_fno_v2_rollout_surface_rollout.gif`
- `results/figures/m13_navier_stokes2d_3d/navier_stokes2d_medium_sm_fno_v2_rollout_velocity_traces_step18.png`

Artifact dimensions:

- Static 3D-style surface PNG: `1304 x 448`.
- Rollout GIF: `1620 x 552`, 18 frames.
- Velocity trace PNG: `1607 x 497`.

These files are generated outputs under `results/figures/` and remain ignored
by git.

## Visualization Source

The generated artifacts use:

- `outputs/navier_stokes2d_medium_sm_fno_v2_diagnostic/predictions.npz`

That artifact contains:

- `inputs`: `(10, 24, 24, 1)`
- `targets`: `(1, 24, 24, 1)`
- `predictions`: `(1, 24, 24, 1)`
- `rollout_targets`: `(36, 24, 24, 1)`
- `rollout_predictions`: `(36, 24, 24, 1)`
- `rollout_relative_l2_per_timestep`: `(36,)`

## Limitations

- The surface height is scalar vorticity value, not physical vertical
  displacement.
- The data is 2D periodic vorticity, not a 3D fluid volume.
- The animation is a visualization of rollout snapshots, not a physically
  rendered fluid simulation.
- The optional velocity traces use a simple periodic streamfunction
  reconstruction and are intended for visual intuition only.
- The current output is derived from one medium diagnostic artifact. Additional
  model variants can be rendered with the same script when needed.

## Reuse

Example command for another prediction artifact:

```bash
PYTHONPATH=src python3 scripts/visualize_navier_stokes2d_3d.py \
  --prediction-path outputs/navier_stokes2d_medium_fno_diagnostic/predictions.npz \
  --output-dir results/figures/m13_navier_stokes2d_3d \
  --prefix navier_stokes2d_medium_fno \
  --surface-step 17 \
  --max-animation-frames 18 \
  --fps 6 \
  --dpi 120 \
  --velocity-traces
```
