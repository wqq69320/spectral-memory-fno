"""Dataset wrappers for PDE trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryBatch(NamedTuple):
    """Container for one input-target trajectory pair."""

    inputs: torch.Tensor
    targets: torch.Tensor


class PDEDataset(Dataset[TrajectoryBatch]):
    """Dataset of fixed-window PDE forecasting pairs.

    Expected trajectory shape is ``(samples, time, *spatial_dims, channels)``.
    Each item returns:

    - inputs with shape ``(input_steps, *spatial_dims, channels)``
    - targets with shape ``(pred_steps, *spatial_dims, channels)``
    """

    def __init__(
        self,
        trajectories: torch.Tensor | np.ndarray,
        input_steps: int = 1,
        pred_steps: int = 1,
    ) -> None:
        """Initialize the dataset."""
        if isinstance(trajectories, np.ndarray):
            trajectories = torch.from_numpy(trajectories)
        if trajectories.ndim < 4:
            raise ValueError("trajectories must have shape (samples, time, ..., channels).")
        if input_steps < 1:
            raise ValueError("input_steps must be at least 1.")
        if pred_steps < 1:
            raise ValueError("pred_steps must be at least 1.")
        if trajectories.shape[1] < input_steps + pred_steps:
            raise ValueError("time dimension must be at least input_steps + pred_steps.")

        self.trajectories = trajectories.float()
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.samples = trajectories.shape[0]
        self.windows_per_sample = trajectories.shape[1] - input_steps - pred_steps + 1

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        input_steps: int = 1,
        pred_steps: int = 1,
        key: str = "trajectories",
    ) -> PDEDataset:
        """Load trajectories from an ``.npz`` file and create a dataset."""
        npz_path = Path(path)
        with np.load(npz_path) as archive:
            if key not in archive:
                raise KeyError(f"Missing '{key}' array in {npz_path}.")
            trajectories = archive[key]
        return cls(trajectories, input_steps=input_steps, pred_steps=pred_steps)

    @property
    def grid_size(self) -> int:
        """Return the final spatial grid size for 1D compatibility callers."""
        return int(self.trajectories.shape[-2])

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return all spatial dimensions between time and channels."""
        return tuple(int(size) for size in self.trajectories.shape[2:-1])

    @property
    def channels(self) -> int:
        """Return the number of field channels."""
        return int(self.trajectories.shape[-1])

    def __len__(self) -> int:
        """Return the number of fixed-window forecasting examples."""
        return self.samples * self.windows_per_sample

    def __getitem__(self, index: int) -> TrajectoryBatch:
        """Return one input window and target window."""
        sample_index = index // self.windows_per_sample
        time_index = index % self.windows_per_sample
        inputs = self.trajectories[sample_index, time_index : time_index + self.input_steps]
        target_start = time_index + self.input_steps
        targets = self.trajectories[sample_index, target_start : target_start + self.pred_steps]
        return TrajectoryBatch(inputs=inputs, targets=targets)
