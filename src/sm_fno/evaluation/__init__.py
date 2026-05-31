"""Evaluation metrics and rollout utilities."""

from __future__ import annotations

from sm_fno.evaluation.costs import count_parameters, inference_timing_summary, seconds_per_unit
from sm_fno.evaluation.dynamic_skill import (
    delta_prediction_relative_l2,
    per_timestep_relative_l2 as per_timestep_dynamic_relative_l2,
    persistence_forecast,
    persistence_skill_score,
    true_step_change_relative_l2,
)
from sm_fno.evaluation.metrics import (
    energy_error,
    fourier_spectrum_error,
    mean_squared_error,
    per_timestep_relative_l2_error,
    relative_l2_error,
)
from sm_fno.evaluation.rollout import autoregressive_rollout

__all__ = [
    "autoregressive_rollout",
    "count_parameters",
    "delta_prediction_relative_l2",
    "energy_error",
    "fourier_spectrum_error",
    "inference_timing_summary",
    "mean_squared_error",
    "per_timestep_dynamic_relative_l2",
    "persistence_forecast",
    "persistence_skill_score",
    "per_timestep_relative_l2_error",
    "relative_l2_error",
    "seconds_per_unit",
    "true_step_change_relative_l2",
]
