"""Tests for Navier-Stokes2D 3D-style visualization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from sm_fno.visualization.navier_stokes2d_3d import (
    coerce_vorticity_sequence,
    frame_indices,
    integrate_particle_traces,
    save_vorticity_surface_comparison_animation,
    vorticity_to_velocity,
)


def test_coerce_vorticity_sequence_accepts_channel_last_rollout() -> None:
    """Prediction artifacts should coerce to time-major scalar vorticity."""
    sequence = np.zeros((4, 6, 5, 1), dtype=np.float32)

    coerced = coerce_vorticity_sequence(sequence)

    assert coerced.shape == (4, 6, 5)
    assert coerced.dtype == np.float32


def test_frame_indices_are_evenly_spaced_and_valid() -> None:
    """Animation frame selection should include valid endpoints."""
    assert frame_indices(5, max_frames=10) == [0, 1, 2, 3, 4]
    assert frame_indices(10, max_frames=3) == [0, 4, 9]
    with pytest.raises(ValueError):
        frame_indices(0)


def test_vorticity_to_velocity_returns_finite_incompressible_field() -> None:
    """Spectral velocity reconstruction should produce finite velocity components."""
    coords = np.linspace(0.0, 1.0, 16, endpoint=False)
    y, x = np.meshgrid(coords, coords, indexing="ij")
    vorticity = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

    velocity_x, velocity_y = vorticity_to_velocity(vorticity)

    assert velocity_x.shape == vorticity.shape
    assert velocity_y.shape == vorticity.shape
    assert np.isfinite(velocity_x).all()
    assert np.isfinite(velocity_y).all()


def test_particle_trace_integration_shape_and_bounds() -> None:
    """Static velocity trace integration should stay in normalized periodic bounds."""
    velocity_x = np.ones((8, 8), dtype=np.float32) * 0.1
    velocity_y = np.zeros((8, 8), dtype=np.float32)

    traces = integrate_particle_traces(velocity_x, velocity_y, particles_per_axis=3, steps=5)

    assert traces.shape == (9, 6, 2)
    assert np.all(traces >= 0.0)
    assert np.all(traces < 1.0)


def test_surface_comparison_animation_requires_matching_shapes(tmp_path) -> None:
    """3D-style comparison animations should reject mismatched rollouts."""
    sequences = {
        "a": np.zeros((2, 4, 4), dtype=np.float32),
        "b": np.zeros((3, 4, 4), dtype=np.float32),
    }

    with pytest.raises(ValueError):
        save_vorticity_surface_comparison_animation(sequences, tmp_path / "bad.gif")
