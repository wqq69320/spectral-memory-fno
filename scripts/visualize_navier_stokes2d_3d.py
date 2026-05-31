"""Render 3D-style diagnostics from Navier-Stokes2D prediction artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sm_fno.visualization.navier_stokes2d_3d import (
    coerce_vorticity_sequence,
    plot_velocity_trace_diagnostic,
    plot_vorticity_surface_triptych,
    save_rollout_surface_animation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Render 3D-style surface and rollout diagnostics from 2D "
            "Navier-Stokes vorticity predictions."
        )
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        required=True,
        help="Path to a predictions.npz artifact from scripts/evaluate.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "figures" / "m13_navier_stokes2d_3d",
        help="Directory for generated visualization artifacts.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix. Defaults to the prediction artifact parent directory name.",
    )
    parser.add_argument(
        "--surface-step",
        type=int,
        default=-1,
        help="Rollout step index for the static surface and trace diagnostics.",
    )
    parser.add_argument(
        "--max-animation-frames",
        type=int,
        default=36,
        help="Maximum number of frames to include in the rollout GIF.",
    )
    parser.add_argument("--fps", type=int, default=6, help="Frames per second for the GIF.")
    parser.add_argument("--dpi", type=int, default=140, help="DPI for generated figures.")
    parser.add_argument(
        "--velocity-traces",
        action="store_true",
        help=(
            "Also render a vorticity-derived 2D incompressible velocity trace "
            "diagnostic for the selected rollout step."
        ),
    )
    return parser.parse_args()


def _resolve_step(step_index: int, sequence_length: int) -> int:
    """Resolve a possibly negative frame index."""
    resolved = step_index + sequence_length if step_index < 0 else step_index
    if resolved < 0 or resolved >= sequence_length:
        raise IndexError(
            f"surface step {step_index} is out of range for sequence length {sequence_length}."
        )
    return resolved


def main() -> None:
    """Render the configured 3D-style diagnostics."""
    args = parse_args()
    prefix = args.prefix or args.prediction_path.parent.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.prediction_path) as archive:
        if "rollout_targets" in archive and "rollout_predictions" in archive:
            target_sequence = coerce_vorticity_sequence(archive["rollout_targets"])
            prediction_sequence = coerce_vorticity_sequence(archive["rollout_predictions"])
            sequence_label = "rollout"
        else:
            target_sequence = coerce_vorticity_sequence(archive["targets"])
            prediction_sequence = coerce_vorticity_sequence(archive["predictions"])
            sequence_label = "one_step"

    step_index = _resolve_step(args.surface_step, target_sequence.shape[0])
    surface_path = args.output_dir / f"{prefix}_{sequence_label}_surface_step{step_index + 1}.png"
    animation_path = args.output_dir / f"{prefix}_{sequence_label}_surface_rollout.gif"
    trace_path = args.output_dir / f"{prefix}_{sequence_label}_velocity_traces_step{step_index + 1}.png"

    plot_vorticity_surface_triptych(
        target_sequence[step_index],
        prediction_sequence[step_index],
        surface_path,
        title=(
            "2D Navier-Stokes vorticity rendered as 3D-style surfaces "
            f"({prefix}, step {step_index + 1})"
        ),
        dpi=args.dpi,
    )
    print(f"[visualize_navier_stokes2d_3d] wrote {surface_path}")

    save_rollout_surface_animation(
        target_sequence,
        prediction_sequence,
        animation_path,
        max_frames=args.max_animation_frames,
        fps=args.fps,
        dpi=args.dpi,
        title=f"2D Navier-Stokes vorticity rollout ({prefix})",
    )
    print(f"[visualize_navier_stokes2d_3d] wrote {animation_path}")

    if args.velocity_traces:
        plot_velocity_trace_diagnostic(
            target_sequence[step_index],
            prediction_sequence[step_index],
            trace_path,
            dpi=args.dpi,
        )
        print(f"[visualize_navier_stokes2d_3d] wrote {trace_path}")


if __name__ == "__main__":
    main()
