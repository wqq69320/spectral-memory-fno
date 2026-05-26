"""Tests for M6 protocol hardening utilities."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(script_name: str) -> ModuleType:
    """Load a script module for direct helper-function tests."""
    script_path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repeated_seed_config_expansion_uses_clear_artifact_paths() -> None:
    """Per-seed config expansion should not mutate the base config."""
    module = _load_script_module("run_repeated_seeds.py")
    base_config = {
        "name": "heat1d_fno_smoke",
        "data_config": "configs/data/heat1d.yaml",
        "model_config": "configs/model/fno1d.yaml",
        "train_config": "configs/train/default.yaml",
        "seed": 42,
    }

    expanded = module.build_seed_experiment_config(base_config, seed=7)

    assert base_config["seed"] == 42
    assert expanded["name"] == "heat1d_fno_smoke_seed7"
    assert expanded["base_experiment"] == "heat1d_fno_smoke"
    assert expanded["run_type"] == "repeated_seed"
    assert expanded["repeated_seed"] is True
    assert expanded["seed"] == 7
    assert expanded["checkpoint_path"] == "results/checkpoints/heat1d_fno_smoke_seed7.pt"
    assert (
        expanded["eval_metrics_path"]
        == "results/tables/heat1d_fno_smoke_seed7_eval_metrics.json"
    )
    assert (
        expanded["prediction_path"]
        == "outputs/repeated_seeds/heat1d_fno_smoke/seed_7/predictions.npz"
    )


def test_aggregate_metrics_on_toy_json(tmp_path: Path) -> None:
    """Aggregation should compute mean/std over repeated-seed metric files."""
    module = _load_script_module("aggregate_metrics.py")
    metric_records = [
        {
            "experiment": "toy_model_seed0",
            "base_experiment": "toy_model",
            "run_type": "repeated_seed",
            "seed": 0,
            "mse": 1.0,
            "relative_l2": 2.0,
            "rollout_relative_l2": 3.0,
            "one_step_inference_seconds": 0.1,
            "rollout_inference_seconds": 0.5,
        },
        {
            "experiment": "toy_model_seed1",
            "base_experiment": "toy_model",
            "run_type": "repeated_seed",
            "seed": 1,
            "mse": 3.0,
            "relative_l2": 4.0,
            "rollout_relative_l2": 5.0,
            "one_step_inference_seconds": 0.3,
            "rollout_inference_seconds": 0.7,
        },
    ]
    for index, record in enumerate(metric_records):
        path = tmp_path / f"toy_model_seed{index}_eval_metrics.json"
        path.write_text(json.dumps(record), encoding="utf-8")

    records = module.load_metric_records(str(tmp_path / "*_eval_metrics.json"))
    aggregate = module.aggregate_records(records)

    assert aggregate["notice"]
    assert len(aggregate["groups"]) == 1
    group = aggregate["groups"][0]
    assert group["base_experiment"] == "toy_model"
    assert group["run_type"] == "repeated_seed"
    assert group["count"] == 2
    assert group["seeds"] == [0, 1]
    assert group["metrics"]["mse"]["mean"] == pytest.approx(2.0)
    assert group["metrics"]["mse"]["std"] == pytest.approx(2**0.5)
    assert group["metrics"]["rollout_inference_seconds"]["mean"] == pytest.approx(0.6)

    output_path = tmp_path / "aggregate_metrics.json"
    module.write_aggregate_json(aggregate, output_path)
    module.write_aggregate_markdown(aggregate, output_path)

    assert output_path.exists()
    assert output_path.with_suffix(".md").exists()


def test_transformer_smoke_config_matches_shared_protocol() -> None:
    """Transformer smoke config should use the same Heat1D smoke protocol."""
    config_path = REPO_ROOT / "configs" / "experiment" / "heat1d_transformer_smoke.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["model_config"] == "configs/model/transformer1d.yaml"
    assert config["input_steps"] == 10
    assert config["pred_steps"] == 1
    assert config["rollout_steps"] == 5
    assert config["epochs"] == 2
    assert config["batch_size"] == 64
    assert config["device"] == "cpu"
    assert config["seed"] == 42
    assert config["train_ratio"] == 0.8
    assert config["val_ratio"] == 0.1
