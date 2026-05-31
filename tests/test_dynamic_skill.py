"""Tests for persistence and dynamic-skill diagnostics."""

from __future__ import annotations

import torch
from torch import nn

from sm_fno.evaluation.dynamic_skill import (
    delta_prediction_relative_l2,
    per_timestep_relative_l2,
    persistence_forecast,
    persistence_skill_score,
    true_step_change_relative_l2,
)
from sm_fno.training.trainer import Trainer


def test_persistence_forecast_repeats_last_input_frame() -> None:
    """Persistence should repeat the final input frame at every rollout step."""
    inputs = torch.arange(2 * 3 * 4 * 1, dtype=torch.float32).reshape(2, 3, 4, 1)
    persistence = persistence_forecast(inputs, steps=5)
    assert persistence.shape == (2, 5, 4, 1)
    assert torch.equal(persistence[:, 0], inputs[:, -1])
    assert torch.equal(persistence[:, -1], inputs[:, -1])


def test_dynamic_skill_is_positive_when_model_beats_persistence() -> None:
    """Skill should be positive when model error is lower than persistence error."""
    model_error = torch.tensor([0.1, 0.2])
    persistence_error = torch.tensor([0.2, 0.4])
    skill = persistence_skill_score(model_error, persistence_error)
    assert torch.allclose(skill, torch.tensor([0.5, 0.5]))


def test_delta_prediction_error_zero_for_exact_delta_prediction() -> None:
    """Delta error should be zero when prediction matches target."""
    inputs = torch.zeros(1, 2, 4, 1)
    targets = torch.ones(1, 3, 4, 1)
    predictions = targets.clone()
    reference = inputs[:, -1]
    error = delta_prediction_relative_l2(predictions, targets, reference)
    assert torch.allclose(error, torch.zeros(3))


def test_true_step_change_relative_l2_shape() -> None:
    """True temporal change should return one value per target step."""
    inputs = torch.zeros(2, 2, 4, 1)
    targets = torch.ones(2, 3, 4, 1)
    change = true_step_change_relative_l2(inputs, targets)
    assert change.shape == (3,)


def test_per_timestep_relative_l2_zero_for_identical_rollouts() -> None:
    """Per-step relative L2 should be zero for identical tensors."""
    target = torch.randn(2, 3, 4, 1)
    assert torch.allclose(per_timestep_relative_l2(target, target), torch.zeros(3))


def test_per_timestep_relative_l2_aggregates_full_batch() -> None:
    """Full-batch dynamic skill should average over held-out trajectories."""
    target = torch.ones(2, 3, 4, 1)
    prediction = target.clone()
    prediction[0, 0] = 2.0
    error = per_timestep_relative_l2(prediction, target)
    assert error.shape == (3,)
    assert error[0] > error[1]
    assert torch.allclose(error[1:], torch.zeros(2))


def test_trainer_delta_and_persistence_loss_options_are_finite() -> None:
    """Trainer loss options should compose with ordinary prediction MSE."""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        delta_loss_weight=0.2,
        persistence_loss_weight=0.1,
    )
    inputs = torch.zeros(2, 2, 4, 1)
    targets = torch.ones(2, 1, 4, 1)
    predictions = torch.full_like(targets, 0.5)
    loss = trainer._compute_loss(inputs, targets, predictions)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
