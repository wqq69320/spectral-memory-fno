"""Tests for M9 cost-efficiency analysis utilities."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

from sm_fno.evaluation.costs import count_parameters, inference_timing_summary


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


def test_parameter_count_and_timing_helpers() -> None:
    """Cost helpers should report deterministic parameter and timing summaries."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2, bias=False))
    counts = count_parameters(model)
    timing = inference_timing_summary(seconds=2.0, examples=4, steps=5)

    assert counts["model_parameter_count"] == 24
    assert counts["model_trainable_parameter_count"] == 24
    assert counts["model_frozen_parameter_count"] == 0
    assert timing["inference_examples_per_second"] == pytest.approx(2.0)
    assert timing["inference_seconds_per_example"] == pytest.approx(0.5)
    assert timing["inference_seconds_per_step"] == pytest.approx(0.4)
    assert timing["inference_seconds_per_example_per_step"] == pytest.approx(0.1)


def test_horizon_sweep_config_expansion_uses_shared_checkpoint() -> None:
    """M9 horizon configs should reuse one per-seed checkpoint across horizons."""
    module = _load_script_module("run_horizon_sweep.py")
    base_config = {
        "name": "heat1d_fno_smoke",
        "data_config": "configs/data/heat1d.yaml",
        "model_config": "configs/model/fno1d.yaml",
        "train_config": "configs/train/default.yaml",
        "rollout_steps": 5,
        "seed": 42,
    }

    train_config = module.build_seed_train_config(
        base_config,
        seed=7,
        analysis_name="m9_cost_efficiency",
    )
    eval_config = module.build_horizon_eval_config(
        base_config,
        seed=7,
        horizon=20,
        train_config=train_config,
        analysis_name="m9_cost_efficiency",
    )

    assert base_config["seed"] == 42
    assert base_config["rollout_steps"] == 5
    assert train_config["name"] == "heat1d_fno_smoke_m9_seed7"
    assert train_config["checkpoint_path"] == "results/checkpoints/heat1d_fno_smoke_m9_seed7.pt"
    assert eval_config["name"] == "heat1d_fno_smoke_h20_seed7"
    assert eval_config["checkpoint_path"] == train_config["checkpoint_path"]
    assert eval_config["rollout_steps"] == 20
    assert eval_config["run_type"] == "horizon_sweep"
    assert (
        eval_config["eval_metrics_path"]
        == "results/tables/m9_cost_efficiency/heat1d_fno_smoke_h20_seed7_eval_metrics.json"
    )

    m11_train_config = module.build_seed_train_config(
        base_config,
        seed=3,
        analysis_name="m11_navier_stokes2d_v2_cost",
    )
    assert m11_train_config["name"] == "heat1d_fno_smoke_m11_seed3"
    assert (
        m11_train_config["checkpoint_path"]
        == "results/checkpoints/heat1d_fno_smoke_m11_seed3.pt"
    )


def test_cost_aggregation_on_toy_json(tmp_path: Path) -> None:
    """Cost aggregation should summarize repeated seeds by horizon."""
    module = _load_script_module("aggregate_cost_metrics.py")
    metric_records = [
        {
            "experiment": "toy_h20_seed0",
            "base_experiment": "toy",
            "run_type": "horizon_sweep",
            "seed": 0,
            "dataset_name": "heat1d",
            "model_name": "fno1d",
            "rollout_steps": 20,
            "mse": 1.0,
            "relative_l2": 2.0,
            "rollout_relative_l2": 3.0,
            "rollout_inference_seconds": 0.4,
            "rollout_seconds_per_step": 0.02,
            "rollout_seconds_per_example": 0.1,
            "rollout_seconds_per_example_per_step": 0.005,
            "one_step_inference_seconds": 0.1,
            "one_step_seconds_per_forward": 0.05,
            "one_step_seconds_per_example": 0.025,
            "model_parameter_count": 100,
            "model_trainable_parameter_count": 100,
        },
        {
            "experiment": "toy_h20_seed1",
            "base_experiment": "toy",
            "run_type": "horizon_sweep",
            "seed": 1,
            "dataset_name": "heat1d",
            "model_name": "fno1d",
            "rollout_steps": 20,
            "mse": 3.0,
            "relative_l2": 4.0,
            "rollout_relative_l2": 5.0,
            "rollout_inference_seconds": 0.8,
            "rollout_seconds_per_step": 0.04,
            "rollout_seconds_per_example": 0.2,
            "rollout_seconds_per_example_per_step": 0.01,
            "one_step_inference_seconds": 0.2,
            "one_step_seconds_per_forward": 0.1,
            "one_step_seconds_per_example": 0.05,
            "model_parameter_count": 100,
            "model_trainable_parameter_count": 100,
        },
    ]
    for index, record in enumerate(metric_records):
        path = tmp_path / f"toy_h20_seed{index}_eval_metrics.json"
        path.write_text(json.dumps(record), encoding="utf-8")

    records = module.load_cost_metric_records(str(tmp_path / "*_eval_metrics.json"))
    aggregate = module.aggregate_cost_records(records)

    assert aggregate["notice"]
    assert len(aggregate["groups"]) == 1
    group = aggregate["groups"][0]
    assert group["dataset_name"] == "heat1d"
    assert group["model_name"] == "fno1d"
    assert group["rollout_steps"] == 20
    assert group["seeds"] == [0, 1]
    assert group["metrics"]["rollout_relative_l2"]["mean"] == pytest.approx(4.0)
    assert group["metrics"]["rollout_seconds_per_step"]["mean"] == pytest.approx(0.03)
    assert group["metrics"]["model_parameter_count"]["mean"] == pytest.approx(100.0)

    output_path = tmp_path / "m9_cost_efficiency_aggregate.json"
    module.write_cost_aggregate_json(aggregate, output_path)
    module.write_cost_aggregate_markdown(aggregate, output_path)

    assert output_path.exists()
    assert output_path.with_suffix(".md").exists()


def test_cost_plot_and_report_helpers(tmp_path: Path) -> None:
    """Plot and generated-report helpers should write protocol artifacts."""
    plot_module = _load_script_module("plot_cost_efficiency.py")
    report_module = _load_script_module("generate_cost_efficiency_report.py")
    aggregate = {
        "notice": "toy notice",
        "groups": [
            {
                "dataset_name": "heat1d",
                "model_name": "fno1d",
                "base_experiment": "heat1d_fno_smoke",
                "run_type": "horizon_sweep",
                "rollout_steps": 5,
                "count": 1,
                "seeds": [0],
                "metrics": {
                    "model_parameter_count": {"mean": 100.0, "std": 0.0},
                    "rollout_relative_l2": {"mean": 0.2, "std": 0.0},
                    "rollout_seconds_per_step": {"mean": 0.01, "std": 0.0},
                },
            }
        ],
    }

    plot_paths = plot_module.write_plots(aggregate, tmp_path / "figures")
    report = report_module.render_cost_efficiency_report(aggregate)

    assert all(path.exists() for path in plot_paths)
    assert "protocol-scale" in report
    assert "does not rank models" in report
