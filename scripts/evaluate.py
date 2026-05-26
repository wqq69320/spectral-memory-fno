"""Evaluate a trained model from a plain YAML experiment config."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sm_fno.evaluation import (
    autoregressive_rollout,
    count_parameters,
    inference_timing_summary,
    mean_squared_error,
    per_timestep_relative_l2_error,
    relative_l2_error,
)
from sm_fno.utils.config import load_yaml
from sm_fno.utils.device import resolve_device
from train import build_model, build_trajectory_datasets, merge_model_config, resolve_data_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate an SM-FNO experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate the configured checkpoint on the held-out trajectory split."""
    args = parse_args()
    experiment_config = load_yaml(args.config)
    data_config = load_yaml(Path(str(experiment_config["data_config"])))
    model_config = merge_model_config(
        experiment_config,
        load_yaml(Path(str(experiment_config["model_config"]))),
    )
    train_config = load_yaml(Path(str(experiment_config["train_config"])))

    experiment_name = str(experiment_config.get("name", "experiment"))
    input_steps = int(experiment_config.get("input_steps", 10))
    pred_steps = int(experiment_config.get("pred_steps", 1))
    seed = int(experiment_config.get("seed", train_config.get("seed", 42)))
    data_path = resolve_data_path(experiment_config, data_config)

    _, _, test_dataset = build_trajectory_datasets(
        data_path,
        input_steps=input_steps,
        pred_steps=pred_steps,
        train_ratio=float(
            experiment_config.get("train_ratio", data_config.get("train_ratio", 0.8))
        ),
        val_ratio=float(experiment_config.get("val_ratio", data_config.get("val_ratio", 0.1))),
        seed=seed,
    )
    batch_size = int(experiment_config.get("batch_size", train_config.get("batch_size", 32)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = resolve_device(str(experiment_config.get("device", train_config.get("device", "cpu"))))
    model = build_model(
        model_config,
        input_steps=input_steps,
        pred_steps=pred_steps,
        grid_size=test_dataset.grid_size,
    )
    checkpoint_path = Path(
        str(
            experiment_config.get(
                "checkpoint_path",
                Path("results") / "checkpoints" / f"{experiment_name}.pt",
            )
        )
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    parameter_counts = count_parameters(model)

    total_examples = 0
    total_mse = 0.0
    total_relative_l2 = 0.0
    forward_passes = 0
    one_step_inference_seconds = 0.0
    sample_inputs: np.ndarray | None = None
    sample_targets: np.ndarray | None = None
    sample_predictions: np.ndarray | None = None

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.inputs.to(device)
            targets = batch.targets.to(device)
            start_time = time.perf_counter()
            predictions = model(inputs)
            one_step_inference_seconds += time.perf_counter() - start_time
            forward_passes += 1
            batch_size = inputs.shape[0]
            total_mse += float(mean_squared_error(predictions, targets).cpu()) * batch_size
            total_relative_l2 += float(relative_l2_error(predictions, targets).cpu()) * batch_size
            total_examples += batch_size

            if sample_inputs is None:
                sample_inputs = inputs[0].cpu().numpy()
                sample_targets = targets[0].cpu().numpy()
                sample_predictions = predictions[0].cpu().numpy()

    if total_examples == 0:
        raise RuntimeError("No test examples were available for evaluation.")

    max_rollout_steps = max(0, test_dataset.trajectories.shape[1] - input_steps)
    requested_rollout_steps = int(experiment_config.get("rollout_steps", pred_steps))
    rollout_steps = min(requested_rollout_steps, max_rollout_steps)
    rollout_metrics: dict[str, object] = {
        "rollout_steps": rollout_steps,
        "rollout_mse": None,
        "rollout_relative_l2": None,
        "rollout_relative_l2_per_timestep": [],
        "rollout_inference_seconds": None,
        "rollout_examples_per_second": None,
        "rollout_seconds_per_step": None,
        "rollout_seconds_per_example": None,
        "rollout_seconds_per_example_per_step": None,
    }
    rollout_predictions_np: np.ndarray | None = None
    rollout_targets_np: np.ndarray | None = None
    rollout_per_timestep_np: np.ndarray | None = None
    if rollout_steps > 0:
        rollout_inputs = test_dataset.trajectories[:, :input_steps].to(device)
        rollout_targets = test_dataset.trajectories[
            :,
            input_steps : input_steps + rollout_steps,
        ].to(device)
        start_time = time.perf_counter()
        rollout_predictions = autoregressive_rollout(model, rollout_inputs, rollout_steps)
        rollout_inference_seconds = time.perf_counter() - start_time
        rollout_per_timestep = per_timestep_relative_l2_error(
            rollout_predictions,
            rollout_targets,
        )
        rollout_metrics = {
            "rollout_steps": rollout_steps,
            "rollout_mse": float(mean_squared_error(rollout_predictions, rollout_targets).cpu()),
            "rollout_relative_l2": float(
                relative_l2_error(rollout_predictions, rollout_targets).cpu()
            ),
            "rollout_relative_l2_per_timestep": [
                float(value) for value in rollout_per_timestep.cpu()
            ],
            "rollout_inference_seconds": rollout_inference_seconds,
            "rollout_examples_per_second": (
                float(rollout_inputs.shape[0] / rollout_inference_seconds)
                if rollout_inference_seconds > 0.0
                else None
            ),
            "rollout_seconds_per_step": (
                float(rollout_inference_seconds / rollout_steps) if rollout_steps > 0 else None
            ),
            "rollout_seconds_per_example": (
                float(rollout_inference_seconds / rollout_inputs.shape[0])
                if rollout_inputs.shape[0] > 0
                else None
            ),
            "rollout_seconds_per_example_per_step": (
                float(rollout_inference_seconds / (rollout_inputs.shape[0] * rollout_steps))
                if rollout_inputs.shape[0] > 0 and rollout_steps > 0
                else None
            ),
        }
        rollout_predictions_np = rollout_predictions[0].cpu().numpy()
        rollout_targets_np = rollout_targets[0].cpu().numpy()
        rollout_per_timestep_np = rollout_per_timestep.cpu().numpy()

    metrics = {
        "experiment": experiment_name,
        "base_experiment": str(experiment_config.get("base_experiment", experiment_name)),
        "analysis_name": str(experiment_config.get("analysis_name", "")),
        "run_type": str(
            experiment_config.get(
                "run_type",
                "repeated_seed" if bool(experiment_config.get("repeated_seed", False)) else "single",
            )
        ),
        "seed": seed,
        "dataset_name": str(data_config.get("name", "unknown")),
        "model_name": str(model_config.get("name", "unknown")),
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(data_path),
        "num_test_examples": total_examples,
        "input_steps": input_steps,
        "pred_steps": pred_steps,
        **parameter_counts,
        "mse": total_mse / total_examples,
        "relative_l2": total_relative_l2 / total_examples,
        "one_step_inference_seconds": one_step_inference_seconds,
        "one_step_forward_passes": forward_passes,
        "one_step_examples_per_second": (
            float(total_examples / one_step_inference_seconds)
            if one_step_inference_seconds > 0.0
            else None
        ),
        "one_step_seconds_per_forward": (
            float(one_step_inference_seconds / forward_passes) if forward_passes > 0 else None
        ),
        "one_step_seconds_per_example": inference_timing_summary(
            seconds=one_step_inference_seconds,
            examples=total_examples,
        )["inference_seconds_per_example"],
        **rollout_metrics,
    }

    metrics_path = Path(
        str(
            experiment_config.get(
                "eval_metrics_path",
                Path("results") / "tables" / f"{experiment_name}_eval_metrics.json",
            )
        )
    )
    prediction_path = Path(
        str(
            experiment_config.get(
                "prediction_path",
                Path("outputs") / experiment_name / "predictions.npz",
            )
        )
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    if sample_inputs is None or sample_targets is None or sample_predictions is None:
        raise RuntimeError("Evaluation did not capture a sample prediction.")
    prediction_payload = {
        "inputs": sample_inputs,
        "targets": sample_targets,
        "predictions": sample_predictions,
    }
    if rollout_targets_np is not None and rollout_predictions_np is not None:
        prediction_payload["rollout_targets"] = rollout_targets_np
        prediction_payload["rollout_predictions"] = rollout_predictions_np
    if rollout_per_timestep_np is not None:
        prediction_payload["rollout_relative_l2_per_timestep"] = rollout_per_timestep_np
    np.savez_compressed(prediction_path, **prediction_payload)

    print(f"[evaluate] saved metrics {metrics_path}")
    print(f"[evaluate] saved sample predictions {prediction_path}")
    print(f"[evaluate] mse {metrics['mse']:.6f}")
    print(f"[evaluate] relative_l2 {metrics['relative_l2']:.6f}")
    print(f"[evaluate] one_step_inference_seconds {one_step_inference_seconds:.6f}")
    if metrics["rollout_relative_l2"] is not None:
        print(f"[evaluate] rollout_relative_l2 {metrics['rollout_relative_l2']:.6f}")
        print(f"[evaluate] rollout_inference_seconds {metrics['rollout_inference_seconds']:.6f}")


if __name__ == "__main__":
    main()
