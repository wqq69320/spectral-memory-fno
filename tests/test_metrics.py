"""Tests for evaluation metrics."""

from __future__ import annotations

import torch

from sm_fno.evaluation.metrics import (
    energy_error,
    fourier_spectrum_error,
    mean_squared_error,
    per_timestep_relative_l2_error,
    relative_l2_error,
)


def test_relative_l2_error_zero_for_identical_tensors() -> None:
    """Identical predictions should have zero relative L2 error."""
    target = torch.ones(2, 8, 1)
    assert torch.isclose(relative_l2_error(target, target), torch.tensor(0.0))


def test_mean_squared_error_zero_for_identical_tensors() -> None:
    """Identical predictions should have zero MSE."""
    target = torch.ones(2, 8, 1)
    assert torch.isclose(mean_squared_error(target, target), torch.tensor(0.0))


def test_energy_error_zero_for_identical_tensors() -> None:
    """Identical predictions should have zero energy error."""
    target = torch.randn(2, 8, 1)
    assert torch.isclose(energy_error(target, target), torch.tensor(0.0))


def test_fourier_spectrum_error_zero_for_identical_tensors() -> None:
    """Identical predictions should have zero spectrum error."""
    target = torch.randn(2, 8, 1)
    assert torch.isclose(fourier_spectrum_error(target, target), torch.tensor(0.0))


def test_per_timestep_relative_l2_error_shape() -> None:
    """Per-timestep rollout errors should return one scalar per forecast step."""
    target = torch.randn(2, 4, 8, 1)
    errors = per_timestep_relative_l2_error(target, target)
    assert errors.shape == (4,)
    assert torch.allclose(errors, torch.zeros(4))
