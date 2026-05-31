"""Run persistence and dynamic-skill diagnostics on saved rollout artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "sm_fno_plot_cache"
_MPL_CONFIG_DIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_DIR = _CACHE_ROOT / "xdg"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sm_fno.evaluation.dynamic_skill import (  # noqa: E402
    delta_prediction_relative_l2,
    per_timestep_relative_l2,
    persistence_forecast,
    persistence_skill_score,
    true_step_change_relative_l2,
)
from sm_fno.utils.config import load_yaml  # noqa: E402
from sm_fno.visualization.navier_stokes2d_3d import plot_vorticity_surface_triptych  # noqa: E402


DEFAULT_EXPERIMENT_CONFIGS = (
    Path("configs/experiment/navier_stokes2d_medium_fno_diagnostic.yaml"),
    Path("configs/experiment/navier_stokes2d_medium_sm_fno_v2_diagnostic.yaml"),
    Path("configs/experiment/navier_stokes2d_medium_sm_fno_v3_diagnostic.yaml"),
    Path("configs/experiment/navier_stokes2d_medium_transformer_diagnostic.yaml"),
)
MODEL_LABELS = {
    "navier_stokes2d_medium_fno_diagnostic": "FNO2D",
    "navier_stokes2d_medium_sm_fno_v2_diagnostic": "SM-FNO2D v2",
    "navier_stokes2d_medium_sm_fno_v3_diagnostic": "SM-FNO2D v3",
    "navier_stokes2d_medium_transformer_diagnostic": "Transformer2D",
    "navier_stokes2d_dynamic_fno_diagnostic": "FNO2D",
    "navier_stokes2d_dynamic_transformer_diagnostic": "Transformer2D",
    "navier_stokes2d_dynamic_sm_fno_v3_diagnostic": "SM-FNO2D v3",
    "navier_stokes2d_dynamic_sm_fno_v3_skill_optimized": "SM-FNO2D v3 optimized",
}
NOTICE = (
    "Dynamic-skill diagnostics are protocol sanity checks only. "
    "They are not benchmark claims or model rankings."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run M15 dynamic-skill diagnostics.")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        action="append",
        dest="experiment_configs",
        help="Experiment config to include. May be provided multiple times.",
    )
    parser.add_argument("--horizon", type=int, default=36, help="Rollout horizon to analyze.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/tables/m15_dynamic_skill_diagnostics.json"),
        help="Path to write diagnostic JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/tables/m15_dynamic_skill_diagnostics.md"),
        help="Path to write diagnostic Markdown table.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures/m15_dynamic_skill"),
        help="Directory for generated diagnostic figures.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="m15",
        help="Filename prefix for generated figures.",
    )
    parser.add_argument(
        "--title-prefix",
        default="M15",
        help="Title prefix used in generated figures.",
    )
    parser.add_argument(
        "--notice",
        default=NOTICE,
        help="Notice text to store in JSON and Markdown outputs.",
    )
    parser.add_argument(
        "--v3-experiment",
        default=None,
        help="Experiment name to use for the SM-FNO2D v3 surface and interpretation.",
    )
    return parser.parse_args()


def _resolve_path(value: object, default: Path) -> Path:
    """Resolve a path-like config value."""
    return Path(str(value)) if value is not None else default


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON if present."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_data_path(experiment_config: dict[str, Any], data_config: dict[str, Any]) -> Path:
    """Resolve the dataset path from experiment and data configs."""
    if "data_path" in experiment_config:
        return Path(str(experiment_config["data_path"]))
    if "output_path" in data_config:
        return Path(str(data_config["output_path"]))
    dataset_name = str(data_config.get("name", "dataset"))
    output_dir = Path(str(data_config.get("output_dir", Path("data") / "processed" / dataset_name)))
    return output_dir / f"{dataset_name}.npz"


def _as_batched_rollout(values: np.ndarray) -> torch.Tensor:
    """Convert a saved single-sample rollout to batched torch format."""
    tensor = torch.from_numpy(np.asarray(values)).float()
    if tensor.ndim < 4:
        raise ValueError("rollout artifact must have shape (time, *spatial, channels).")
    return tensor.unsqueeze(0)


def _as_rollout_batch(values: np.ndarray) -> torch.Tensor:
    """Convert saved rollout arrays to batched torch format."""
    tensor = torch.from_numpy(np.asarray(values)).float()
    if tensor.ndim < 4:
        raise ValueError("rollout artifact must include time, spatial, and channel dims.")
    if tensor.ndim == 4:
        return tensor.unsqueeze(0)
    return tensor


def _float_list(values: torch.Tensor) -> list[float]:
    """Convert a tensor to a list of Python floats."""
    return [float(value) for value in values.detach().cpu()]


def _series_summary(values: torch.Tensor) -> dict[str, float]:
    """Return mean and final values for a per-step tensor."""
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "final": float(values[-1]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _load_dataset_temporal_diagnostics(config_path: Path, horizon: int) -> dict[str, Any]:
    """Compute persistence and true-change diagnostics on the held-out split."""
    experiment_config = load_yaml(config_path)
    data_config = load_yaml(Path(str(experiment_config["data_config"])))
    train_config = load_yaml(Path(str(experiment_config["train_config"])))
    data_path = _resolve_data_path(experiment_config, data_config)
    seed = int(experiment_config.get("seed", train_config.get("seed", 42)))
    input_steps = int(experiment_config.get("input_steps", 10))
    train_ratio = float(experiment_config.get("train_ratio", data_config.get("train_ratio", 0.8)))
    val_ratio = float(experiment_config.get("val_ratio", data_config.get("val_ratio", 0.1)))

    with np.load(data_path) as archive:
        trajectories = torch.from_numpy(archive["trajectories"]).float()

    num_samples = trajectories.shape[0]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator)
    train_count = max(1, int(num_samples * train_ratio))
    val_count = max(1, int(num_samples * val_ratio))
    if train_count + val_count >= num_samples:
        val_count = max(0, num_samples - train_count - 1)
    test_indices = indices[train_count + val_count :]
    test_trajectories = trajectories[test_indices]
    steps = min(horizon, test_trajectories.shape[1] - input_steps)
    inputs = test_trajectories[:, :input_steps]
    targets = test_trajectories[:, input_steps : input_steps + steps]
    persistence = persistence_forecast(inputs, steps=steps)
    persistence_error = per_timestep_relative_l2(persistence, targets)
    true_step_change = true_step_change_relative_l2(inputs, targets)
    flattened_targets = targets.reshape(targets.shape[0], targets.shape[1], -1)

    return {
        "data_path": str(data_path),
        "num_test_trajectories": int(test_trajectories.shape[0]),
        "horizon": int(steps),
        "persistence_relative_l2_per_timestep": _float_list(persistence_error),
        "true_step_change_relative_l2_per_timestep": _float_list(true_step_change),
        "target_std_per_timestep": [
            float(value) for value in flattened_targets.std(dim=(0, 2), unbiased=False)
        ],
        "persistence_relative_l2": _series_summary(persistence_error),
        "true_step_change_relative_l2": _series_summary(true_step_change),
    }


def _load_record(
    config_path: Path,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Load one experiment artifact and compute dynamic-skill diagnostics."""
    config = load_yaml(config_path)
    experiment_name = str(config.get("name", config_path.stem))
    label = MODEL_LABELS.get(experiment_name, experiment_name)
    prediction_path = _resolve_path(
        config.get("prediction_path"),
        Path("outputs") / experiment_name / "predictions.npz",
    )
    metrics_path = _resolve_path(
        config.get("eval_metrics_path"),
        Path("results/tables") / f"{experiment_name}_eval_metrics.json",
    )
    eval_metrics = _load_json(metrics_path)
    full_test_rollouts_path = _resolve_path(
        config.get("full_test_rollouts_path"),
        Path(str(eval_metrics.get("full_test_rollouts_path")))
        if eval_metrics and eval_metrics.get("full_test_rollouts_path")
        else Path("outputs") / experiment_name / "full_test_rollouts.npz",
    )

    base_record: dict[str, Any] = {
        "experiment": experiment_name,
        "label": label,
        "config_path": str(config_path),
        "prediction_path": str(prediction_path),
        "full_test_rollouts_path": str(full_test_rollouts_path),
        "metrics_path": str(metrics_path),
        "available": full_test_rollouts_path.exists() or prediction_path.exists(),
    }
    if not full_test_rollouts_path.exists() and not prediction_path.exists():
        base_record["skip_reason"] = "prediction artifact missing"
        return base_record, None

    diagnostic_scope = "full_test" if full_test_rollouts_path.exists() else "saved_sample"
    artifact_path = full_test_rollouts_path if full_test_rollouts_path.exists() else prediction_path
    with np.load(artifact_path) as archive:
        if "rollout_targets" not in archive or "rollout_predictions" not in archive:
            base_record["available"] = False
            base_record["skip_reason"] = "rollout targets/predictions missing"
            return base_record, None
        inputs = _as_rollout_batch(archive["inputs"])
        targets = _as_rollout_batch(archive["rollout_targets"])
        predictions = _as_rollout_batch(archive["rollout_predictions"])

    steps = min(horizon, targets.shape[1], predictions.shape[1])
    inputs = inputs.float()
    targets = targets[:, :steps].float()
    predictions = predictions[:, :steps].float()
    persistence = persistence_forecast(inputs, steps=steps)
    reference = inputs[:, -1]

    model_error = per_timestep_relative_l2(predictions, targets)
    persistence_error = per_timestep_relative_l2(persistence, targets)
    true_step_change = true_step_change_relative_l2(inputs, targets)
    delta_error = delta_prediction_relative_l2(predictions, targets, reference)
    skill = persistence_skill_score(model_error, persistence_error)

    reported_rollout = None
    if eval_metrics is not None:
        reported_rollout = eval_metrics.get("rollout_relative_l2_per_timestep")
        if isinstance(reported_rollout, list):
            reported_rollout = reported_rollout[:steps]

    record = {
        **base_record,
        "available": True,
        "diagnostic_scope": diagnostic_scope,
        "artifact_path": str(artifact_path),
        "horizon": steps,
        "reported_rollout_relative_l2": eval_metrics.get("rollout_relative_l2")
        if eval_metrics
        else None,
        "reported_rollout_relative_l2_per_timestep": reported_rollout,
        "sample_model_relative_l2_per_timestep": _float_list(model_error),
        "sample_persistence_relative_l2_per_timestep": _float_list(persistence_error),
        "sample_true_step_change_relative_l2_per_timestep": _float_list(true_step_change),
        "sample_delta_prediction_relative_l2_per_timestep": _float_list(delta_error),
        "sample_persistence_skill_per_timestep": _float_list(skill),
        "sample_model_relative_l2": _series_summary(model_error),
        "sample_persistence_relative_l2": _series_summary(persistence_error),
        "sample_true_step_change_relative_l2": _series_summary(true_step_change),
        "sample_delta_prediction_relative_l2": _series_summary(delta_error),
        "sample_persistence_skill": _series_summary(skill),
        "full_test_model_relative_l2_per_timestep": _float_list(model_error),
        "full_test_persistence_relative_l2_per_timestep": _float_list(persistence_error),
        "full_test_delta_prediction_relative_l2_per_timestep": _float_list(delta_error),
        "full_test_persistence_skill_per_timestep": _float_list(skill),
        "full_test_model_relative_l2": _series_summary(model_error),
        "full_test_persistence_relative_l2": _series_summary(persistence_error),
        "full_test_delta_prediction_relative_l2": _series_summary(delta_error),
        "full_test_persistence_skill": _series_summary(skill),
        "beats_persistence_mean": bool(skill.mean() > 0.0),
        "beats_persistence_final": bool(skill[-1] > 0.0),
    }
    arrays = {
        "inputs": inputs[0].numpy(),
        "targets": targets[0].numpy(),
        "predictions": predictions[0].numpy(),
    }
    return record, arrays


def _plot_rollout_comparison(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    title_prefix: str,
) -> None:
    """Plot model rollout error against persistence and true temporal change."""
    available = [record for record in records if record.get("available")]
    if not available:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, int(available[0]["horizon"]) + 1)
    persistence = available[0]["sample_persistence_relative_l2_per_timestep"]
    true_change = available[0]["sample_true_step_change_relative_l2_per_timestep"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, persistence, color="black", linestyle="--", label="persistence error")
    ax.plot(steps, true_change, color="0.45", linestyle=":", label="true step change")
    for record in available:
        ax.plot(
            steps,
            record["sample_model_relative_l2_per_timestep"],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=record["label"],
        )
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2")
    ax.set_title(f"{title_prefix} rollout error vs persistence and true temporal change")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_dataset_temporal_diagnostics(
    diagnostics: dict[str, Any],
    output_path: Path,
    *,
    title_prefix: str,
) -> None:
    """Plot held-out true trajectory change diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, int(diagnostics["horizon"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        steps,
        diagnostics["persistence_relative_l2_per_timestep"],
        color="black",
        linestyle="--",
        label="persistence error",
    )
    ax.plot(
        steps,
        diagnostics["true_step_change_relative_l2_per_timestep"],
        color="0.45",
        linestyle=":",
        label="true step-to-step change",
    )
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2")
    ax.set_title(f"{title_prefix} held-out true temporal variation diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_series(records: list[dict[str, Any]], key: str, ylabel: str, output_path: Path) -> None:
    """Plot one per-step diagnostic series for each available model."""
    available = [record for record in records if record.get("available")]
    if not available:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, int(available[0]["horizon"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for record in available:
        ax.plot(
            steps,
            record[key],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=record["label"],
        )
    if "skill" in key:
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", label="persistence parity")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_markdown(
    records: list[dict[str, Any]],
    dataset_temporal_diagnostics: dict[str, Any],
    interpretation: str,
    output_path: Path,
    *,
    title_prefix: str,
    notice: str,
) -> None:
    """Write a compact Markdown diagnostic table."""
    scope_label = (
        "Full-Test"
        if any(record.get("diagnostic_scope") == "full_test" for record in records)
        else "Sample"
    )
    lines = [
        f"# {title_prefix} Dynamic Skill Diagnostics",
        "",
        notice,
        "",
        f"| Model | {scope_label} Model Rel L2 Mean | {scope_label} Model Rel L2 Step 36 | "
        "Persistence Rel L2 Mean | Persistence Rel L2 Step 36 | "
        "Skill Mean | Skill Step 36 | Delta Error Mean | Delta Error Step 36 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        if not record.get("available"):
            lines.append(
                f"| {record['label']} | NA | NA | NA | NA | NA | NA | NA | NA |"
            )
            continue
        model = record["sample_model_relative_l2"]
        persistence = record["sample_persistence_relative_l2"]
        skill = record["sample_persistence_skill"]
        delta = record["sample_delta_prediction_relative_l2"]
        lines.append(
            f"| {record['label']} | "
            f"{model['mean']:.6f} | {model['final']:.6f} | "
            f"{persistence['mean']:.6f} | {persistence['final']:.6f} | "
            f"{skill['mean']:.6f} | {skill['final']:.6f} | "
            f"{delta['mean']:.6f} | {delta['final']:.6f} |"
        )
    dataset_persistence = dataset_temporal_diagnostics["persistence_relative_l2"]
    dataset_step_change = dataset_temporal_diagnostics["true_step_change_relative_l2"]
    lines.extend(
        [
            "",
            "## Held-Out True Temporal Variation",
            "",
            "| Diagnostic | Mean | Step 36 |",
            "| --- | ---: | ---: |",
            "| Persistence error / change from last input | "
            f"{dataset_persistence['mean']:.6f} | {dataset_persistence['final']:.6f} |",
            "| True step-to-step change | "
            f"{dataset_step_change['mean']:.6f} | {dataset_step_change['final']:.6f} |",
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _find_v3_experiment(records: list[dict[str, Any]], requested: str | None) -> str | None:
    """Find the v3 experiment name to interpret."""
    if requested:
        return requested
    for record in records:
        experiment = str(record.get("experiment", ""))
        if "sm_fno_v3" in experiment:
            return experiment
    return None


def _interpret_v3(records: list[dict[str, Any]], v3_experiment: str | None) -> str:
    """Return a bounded interpretation for SM-FNO2D v3."""
    if v3_experiment is None:
        return "SM-FNO2D v3 artifact was not configured, so persistence skill was not assessed."
    v3 = next(
        (
            record
            for record in records
            if record.get("experiment") == v3_experiment
        ),
        None,
    )
    if not v3 or not v3.get("available"):
        return "SM-FNO2D v3 artifact was not available, so persistence skill could not be assessed."
    skill = v3["sample_persistence_skill"]
    scope = (
        "full held-out rollouts"
        if v3.get("diagnostic_scope") == "full_test"
        else "saved sample artifact"
    )
    if skill["mean"] > 0.05 and skill["final"] > 0.05:
        return (
            f"SM-FNO2D v3 is better than persistence on the {scope} by the "
            "predeclared positive-skill threshold. This remains a diagnostic result only."
        )
    return (
        f"SM-FNO2D v3 is not meaningfully better than persistence on the {scope}: "
        "mean and/or step-36 persistence-normalized skill is not positive by the "
        "predeclared threshold. Low rollout error must therefore be interpreted alongside "
        "true temporal variation in this diagnostic fixture."
    )


def main() -> None:
    """Run diagnostics, write tables, and generate plots."""
    args = parse_args()
    config_paths = args.experiment_configs or list(DEFAULT_EXPERIMENT_CONFIGS)
    records: list[dict[str, Any]] = []
    arrays_by_experiment: dict[str, dict[str, np.ndarray]] = {}

    for config_path in config_paths:
        record, arrays = _load_record(config_path, args.horizon)
        records.append(record)
        if arrays is not None:
            arrays_by_experiment[record["experiment"]] = arrays

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.artifact_prefix
    comparison_path = args.figures_dir / f"{prefix}_rollout_vs_persistence.png"
    skill_path = args.figures_dir / f"{prefix}_persistence_skill.png"
    delta_path = args.figures_dir / f"{prefix}_delta_prediction_error.png"
    temporal_path = args.figures_dir / f"{prefix}_heldout_temporal_variation.png"
    v3_surface_path = args.figures_dir / f"{prefix}_sm_fno2d_v3_surface_step36.png"
    dataset_temporal_diagnostics = _load_dataset_temporal_diagnostics(config_paths[0], args.horizon)
    v3_experiment = _find_v3_experiment(records, args.v3_experiment)

    _plot_rollout_comparison(records, comparison_path, title_prefix=args.title_prefix)
    _plot_dataset_temporal_diagnostics(
        dataset_temporal_diagnostics,
        temporal_path,
        title_prefix=args.title_prefix,
    )
    _plot_series(
        records,
        "sample_persistence_skill_per_timestep",
        "Persistence-normalized skill",
        skill_path,
    )
    _plot_series(
        records,
        "sample_delta_prediction_relative_l2_per_timestep",
        "Delta-prediction relative L2",
        delta_path,
    )

    v3_arrays = arrays_by_experiment.get(v3_experiment or "")
    if v3_arrays is not None:
        step_index = min(args.horizon, v3_arrays["targets"].shape[0]) - 1
        plot_vorticity_surface_triptych(
            target=v3_arrays["targets"][step_index, :, :, 0],
            prediction=v3_arrays["predictions"][step_index, :, :, 0],
            output_path=v3_surface_path,
            title=f"SM-FNO2D v3 step {step_index + 1} dynamic-skill surface diagnostic",
            dpi=140,
        )

    interpretation = _interpret_v3(records, v3_experiment)
    payload = {
        "notice": args.notice,
        "horizon": args.horizon,
        "records": records,
        "heldout_temporal_diagnostics": dataset_temporal_diagnostics,
        "plots": {
            "rollout_vs_persistence": str(comparison_path),
            "persistence_skill": str(skill_path),
            "delta_prediction_error": str(delta_path),
            "heldout_temporal_variation": str(temporal_path),
            "sm_fno2d_v3_surface_step36": str(v3_surface_path)
            if v3_arrays is not None
            else None,
        },
        "interpretation": interpretation,
    }
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    _write_markdown(
        records,
        dataset_temporal_diagnostics,
        interpretation,
        args.output_md,
        title_prefix=args.title_prefix,
        notice=args.notice,
    )

    print(f"[dynamic_skill] wrote {args.output_json}")
    print(f"[dynamic_skill] wrote {args.output_md}")
    print(f"[dynamic_skill] wrote {comparison_path}")
    print(f"[dynamic_skill] wrote {temporal_path}")
    print(f"[dynamic_skill] wrote {skill_path}")
    print(f"[dynamic_skill] wrote {delta_path}")
    if v3_arrays is not None:
        print(f"[dynamic_skill] wrote {v3_surface_path}")
    print(f"[dynamic_skill] interpretation: {interpretation}")


if __name__ == "__main__":
    main()
