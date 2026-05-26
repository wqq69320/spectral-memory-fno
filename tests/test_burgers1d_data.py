"""Tests for Burgers1D data generation and configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from sm_fno.data.burgers1d import generate_burgers1d


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_burgers1d_generator_shape_and_finite_values() -> None:
    """Default Burgers1D config should return finite [samples, time, grid, channels]."""
    config_path = REPO_ROOT / "configs" / "data" / "burgers1d.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data = generate_burgers1d(
        num_samples=config["num_samples"],
        grid_size=config["grid_size"],
        time_steps=config["time_steps"],
        viscosity=config["viscosity"],
        dt=config["dt"],
        domain_length=config["domain_length"],
        seed=config["seed"],
    )

    assert data.shape == (
        config["num_samples"],
        config["time_steps"],
        config["grid_size"],
        1,
    )
    assert torch.isfinite(data).all()
    assert not torch.allclose(data[:, 0], data[:, -1])


def test_burgers1d_generator_rejects_combined_unstable_step() -> None:
    """Combined explicit stability guard should reject configs before NaNs."""
    with pytest.raises(ValueError, match="combined convection-diffusion stability"):
        generate_burgers1d(
            num_samples=2,
            grid_size=64,
            time_steps=4,
            viscosity=0.01,
            dt=0.01,
            domain_length=1.0,
            seed=7,
        )


def test_burgers1d_smoke_configs_match_shared_protocol() -> None:
    """Burgers smoke configs should share the M6/M7 protocol settings."""
    expected_model_configs = {
        "burgers1d_fno_smoke.yaml": "configs/model/fno1d.yaml",
        "burgers1d_sm_fno_smoke.yaml": "configs/model/sm_fno1d.yaml",
        "burgers1d_transformer_smoke.yaml": "configs/model/transformer1d.yaml",
    }

    for config_name, model_config in expected_model_configs.items():
        config_path = REPO_ROOT / "configs" / "experiment" / config_name
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        assert config["data_config"] == "configs/data/burgers1d.yaml"
        assert config["data_path"] == "data/processed/burgers1d/burgers1d.npz"
        assert config["model_config"] == model_config
        assert config["input_steps"] == 10
        assert config["pred_steps"] == 1
        assert config["rollout_steps"] == 5
        assert config["epochs"] == 2
        assert config["batch_size"] == 64
        assert config["learning_rate"] == 0.001
        assert config["weight_decay"] == 0.0001
        assert config["grad_clip_norm"] == 1.0
        assert config["device"] == "cpu"
        assert config["seed"] == 42
        assert config["train_ratio"] == 0.8
        assert config["val_ratio"] == 0.1
