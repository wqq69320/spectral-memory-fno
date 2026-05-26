"""Tests for Heat1D generation and dataset windows."""

from __future__ import annotations

import numpy as np

from sm_fno.data.datasets import PDEDataset
from sm_fno.data.heat1d import generate_heat1d


def test_heat1d_generator_shape() -> None:
    """Heat1D generator should return [samples, time, grid, channels]."""
    data = generate_heat1d(
        num_samples=4,
        grid_size=16,
        time_steps=12,
        alpha=0.01,
        dt=0.001,
        domain_length=1.0,
        seed=7,
    )
    assert data.shape == (4, 12, 16, 1)


def test_pde_dataset_npz_window_shapes(tmp_path) -> None:
    """NPZ-backed dataset should return fixed input and target windows."""
    trajectories = np.zeros((3, 12, 16, 1), dtype=np.float32)
    data_path = tmp_path / "heat1d.npz"
    np.savez_compressed(data_path, trajectories=trajectories)

    dataset = PDEDataset.from_npz(data_path, input_steps=4, pred_steps=2)
    batch = dataset[0]

    assert batch.inputs.shape == (4, 16, 1)
    assert batch.targets.shape == (2, 16, 1)
