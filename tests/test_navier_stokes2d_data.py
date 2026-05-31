"""Tests for Navier-Stokes2D data generation and configs."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from sm_fno.data.navier_stokes2d import generate_navier_stokes2d


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_navier_stokes2d_generator_shape_and_finite_values() -> None:
    """Default Navier-Stokes2D config should return finite 2D vorticity fields."""
    config_path = REPO_ROOT / "configs" / "data" / "navier_stokes2d.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data = generate_navier_stokes2d(
        num_samples=4,
        grid_size=config["grid_size"],
        time_steps=8,
        viscosity=config["viscosity"],
        dt=config["dt"],
        domain_length=config["domain_length"],
        seed=config["seed"],
        cfl_safety=config["cfl_safety"],
    )

    assert data.shape == (4, 8, config["grid_size"], config["grid_size"], 1)
    assert torch.isfinite(data).all()
    assert not torch.allclose(data[:, 0], data[:, -1])


def test_navier_stokes2d_generator_rejects_unstable_cfl() -> None:
    """The generator should reject clearly unstable explicit advection steps."""
    with torch.no_grad():
        try:
            generate_navier_stokes2d(
                num_samples=2,
                grid_size=16,
                time_steps=4,
                viscosity=0.001,
                dt=10.0,
                domain_length=1.0,
                seed=7,
                cfl_safety=0.01,
            )
        except ValueError as error:
            assert "CFL" in str(error)
        else:
            raise AssertionError("Expected a CFL ValueError for an unstable configuration.")


def test_navier_stokes2d_save_every_changes_saved_frame_spacing() -> None:
    """Saving every internal step should produce smaller changes than sparse saves."""
    common_kwargs = {
        "num_samples": 2,
        "grid_size": 12,
        "time_steps": 6,
        "viscosity": 0.001,
        "dt": 0.002,
        "domain_length": 1.0,
        "seed": 11,
        "cfl_safety": 0.95,
        "initial_amplitude": 1.2,
        "max_modes": 3,
    }
    dense = generate_navier_stokes2d(**common_kwargs, save_every=1)
    sparse = generate_navier_stokes2d(**common_kwargs, save_every=4)

    dense_change = torch.linalg.vector_norm(dense[:, -1] - dense[:, 0])
    sparse_change = torch.linalg.vector_norm(sparse[:, -1] - sparse[:, 0])
    assert sparse_change > dense_change


def test_navier_stokes2d_dynamic_configs_match_m16_protocol() -> None:
    """M16 dynamic configs should target the persistence-hard diagnostic fixture."""
    data_config_path = REPO_ROOT / "configs" / "data" / "navier_stokes2d_dynamic.yaml"
    with data_config_path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    assert data_config["grid_size"] == 24
    assert data_config["time_steps"] >= 46
    assert data_config["save_every"] > 1
    assert data_config["dt"] * data_config["save_every"] > 0.02
    assert data_config["viscosity"] <= 0.001

    expected = {
        "navier_stokes2d_dynamic_fno_diagnostic.yaml": "configs/model/fno2d.yaml",
        "navier_stokes2d_dynamic_sm_fno_v3_diagnostic.yaml": "configs/model/sm_fno2d_v3.yaml",
    }
    for config_name, model_config in expected.items():
        config_path = REPO_ROOT / "configs" / "experiment" / config_name
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        assert config["data_config"] == "configs/data/navier_stokes2d_dynamic.yaml"
        assert config["model_config"] == model_config
        assert config["input_steps"] == 10
        assert config["pred_steps"] == 1
        assert config["rollout_steps"] == 36
        assert config["device"] == "cpu"


def test_navier_stokes2d_smoke_configs_match_shared_protocol() -> None:
    """Navier-Stokes smoke configs should share M10 protocol settings."""
    expected_model_configs = {
        "navier_stokes2d_fno_smoke.yaml": "configs/model/fno2d.yaml",
        "navier_stokes2d_sm_fno_smoke.yaml": "configs/model/sm_fno2d.yaml",
        "navier_stokes2d_sm_fno_v2_smoke.yaml": "configs/model/sm_fno2d_v2.yaml",
        "navier_stokes2d_transformer_smoke.yaml": "configs/model/transformer2d.yaml",
    }

    for config_name, model_config in expected_model_configs.items():
        config_path = REPO_ROOT / "configs" / "experiment" / config_name
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        assert config["data_config"] == "configs/data/navier_stokes2d.yaml"
        assert config["data_path"] == "data/processed/navier_stokes2d/navier_stokes2d.npz"
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


def test_navier_stokes2d_medium_v3_config_uses_rollout_training() -> None:
    """The M14 v3 medium diagnostic config should enable rollout-aware training."""
    config_path = (
        REPO_ROOT
        / "configs"
        / "experiment"
        / "navier_stokes2d_medium_sm_fno_v3_diagnostic.yaml"
    )
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["data_config"] == "configs/data/navier_stokes2d_medium.yaml"
    assert config["model_config"] == "configs/model/sm_fno2d_v3.yaml"
    assert config["input_steps"] == 10
    assert config["pred_steps"] == 1
    assert config["rollout_steps"] == 36
    assert config["rollout_train_steps"] >= 1
    assert config["rollout_loss_weight"] > 0.0
