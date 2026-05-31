# M13 Decision Log

## Decisions Made

- Render 2D vorticity as 3D-style surfaces using Matplotlib `mplot3d`.
- Use GIF output through Pillow for rollout animation to avoid requiring
  system-level video encoders.
- Use the medium SM-FNO2D v2 diagnostic prediction artifact as the first
  presentation source because it has a 24x24 grid and rollout-36 sequence.
- Include true, predicted, and absolute-error panels in both static and
  animated surface outputs.
- Add optional velocity traces derived from the periodic 2D vorticity
  streamfunction solve, and document them as qualitative visualization aids.

## Rationale

- Surface rendering is visually useful for a forum presentation while still
  staying honest about the underlying 2D data.
- GIF output is portable in slides and does not depend on `ffmpeg`.
- Keeping the script artifact-driven lets the same visualization run on smoke,
  medium, FNO2D, SM-FNO2D v2, or Transformer2D prediction files.

## Known Limitations

- The surface z-axis is scalar vorticity, not a third spatial coordinate.
- Velocity traces are derived from one selected snapshot and are not validated
  time-dependent particle trajectories.
- Matplotlib surface animations are presentation diagnostics, not simulation
  products.

## Follow-Up Work

- Render matching FNO2D and Transformer2D animations if the forum deck needs
  side-by-side model diagnostics.
- Add MP4 export if a target presentation environment handles MP4 more
  reliably than GIF.
- Add denser or seeded trace controls if velocity-trace visuals become a
  central slide.
