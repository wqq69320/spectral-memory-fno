"""Train a model from a plain YAML experiment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sm_fno.data.datasets import PDEDataset
from sm_fno.evaluation.costs import count_parameters
from sm_fno.models import (
    FNO1D,
    FNO2D,
    MLPBaseline,
    SpectralMemoryFNO1D,
    SpectralMemoryFNO2D,
    SpectralMemoryFNO2DV2,
    SpectralMemoryFNO2DV3,
    Transformer1DBaseline,
    Transformer2DBaseline,
)
from sm_fno.training import Trainer, build_optimizer
from sm_fno.utils.config import load_yaml
from sm_fno.utils.device import resolve_device
from sm_fno.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an SM-FNO experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment YAML config.",
    )
    return parser.parse_args()


def resolve_data_path(experiment_config: dict[str, object], data_config: dict[str, object]) -> Path:
    """Resolve the dataset path from experiment and data configs."""
    if "data_path" in experiment_config:
        return Path(str(experiment_config["data_path"]))
    if "output_path" in data_config:
        return Path(str(data_config["output_path"]))
    dataset_name = str(data_config.get("name", "dataset"))
    output_dir = Path(str(data_config.get("output_dir", Path("data") / "processed" / dataset_name)))
    return output_dir / f"{dataset_name}.npz"


def merge_model_config(
    experiment_config: dict[str, object],
    model_config: dict[str, object],
) -> dict[str, object]:
    """Apply optional experiment-local model overrides."""
    overrides = experiment_config.get("model_overrides", {})
    if overrides is None:
        return dict(model_config)
    if not isinstance(overrides, dict):
        raise TypeError("model_overrides must be a mapping when provided.")
    merged = dict(model_config)
    merged.update(overrides)
    return merged


def build_trajectory_datasets(
    data_path: Path,
    *,
    input_steps: int,
    pred_steps: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[PDEDataset, PDEDataset | None, PDEDataset]:
    """Split trajectories by sample and return train, validation, and test datasets."""
    with np.load(data_path) as archive:
        trajectories = torch.from_numpy(archive["trajectories"]).float()

    num_samples = trajectories.shape[0]
    if num_samples < 3:
        raise ValueError("Need at least three trajectories for train/val/test splitting.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator)
    train_count = max(1, int(num_samples * train_ratio))
    val_count = max(1, int(num_samples * val_ratio))
    if train_count + val_count >= num_samples:
        val_count = max(0, num_samples - train_count - 1)
    test_count = num_samples - train_count - val_count
    if test_count < 1:
        raise ValueError("Split ratios leave no test trajectories.")

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]

    train_dataset = PDEDataset(
        trajectories[train_idx],
        input_steps=input_steps,
        pred_steps=pred_steps,
    )
    val_dataset = (
        PDEDataset(trajectories[val_idx], input_steps=input_steps, pred_steps=pred_steps)
        if len(val_idx) > 0
        else None
    )
    test_dataset = PDEDataset(
        trajectories[test_idx],
        input_steps=input_steps,
        pred_steps=pred_steps,
    )
    return train_dataset, val_dataset, test_dataset


def build_model(
    model_config: dict[str, object],
    *,
    input_steps: int,
    pred_steps: int,
    grid_size: int,
) -> torch.nn.Module:
    """Build the configured model."""
    model_name = str(model_config.get("name", "mlp")).lower()
    if model_name in {"mlp", "mlpbaseline"}:
        return MLPBaseline(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 1)),
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            num_layers=int(model_config.get("num_layers", 4)),
            dropout=float(model_config.get("dropout", 0.0)),
            input_steps=input_steps,
            pred_steps=pred_steps,
            grid_size=grid_size,
        )
    if model_name in {"fno", "fno1d"}:
        return FNO1D(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 1)),
            modes=int(model_config.get("modes", 16)),
            width=int(model_config.get("width", 64)),
            depth=int(model_config.get("depth", 4)),
            dropout=float(model_config.get("dropout", 0.0)),
            input_steps=input_steps,
            pred_steps=pred_steps,
        )
    if model_name in {"fno2d"}:
        return FNO2D(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 1)),
            modes=int(model_config.get("modes", 8)),
            width=int(model_config.get("width", 32)),
            depth=int(model_config.get("depth", 4)),
            dropout=float(model_config.get("dropout", 0.0)),
            input_steps=input_steps,
            pred_steps=pred_steps,
        )
    if model_name in {"sm_fno", "sm_fno1d", "spectralmemoryfno1d", "spectral_memory_fno1d"}:
        return SpectralMemoryFNO1D(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 1)),
            modes=int(model_config.get("modes", 16)),
            width=int(model_config.get("width", 64)),
            state_dim=int(model_config.get("state_dim", 64)),
            depth=int(model_config.get("depth", 4)),
            dropout=float(model_config.get("dropout", 0.0)),
            input_steps=input_steps,
            pred_steps=pred_steps,
        )
    if model_name in {"sm_fno2d", "spectralmemoryfno2d", "spectral_memory_fno2d"}:
        return SpectralMemoryFNO2D(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 1)),
            modes=int(model_config.get("modes", 8)),
            width=int(model_config.get("width", 32)),
            state_dim=int(model_config.get("state_dim", 32)),
            depth=int(model_config.get("depth", 4)),
            dropout=float(model_config.get("dropout", 0.0)),
            input_steps=input_steps,
            pred_steps=pred_steps,
        )
    if model_name in {"sm_fno2d_v2", "spectralmemoryfno2dv2", "spectral_memory_fno2d_v2"}:
        return SpectralMemoryFNO2DV2(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 1)),
            modes=int(model_config.get("modes", 8)),
            width=int(model_config.get("width", 32)),
            state_dim=int(model_config.get("state_dim", 32)),
            depth=int(model_config.get("depth", 4)),
            dropout=float(model_config.get("dropout", 0.0)),
            input_steps=input_steps,
            pred_steps=pred_steps,
        )
    if model_name in {"sm_fno2d_v3", "spectralmemoryfno2dv3", "spectral_memory_fno2d_v3"}:
        return SpectralMemoryFNO2DV3(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 1)),
            modes=int(model_config.get("modes", 8)),
            width=int(model_config.get("width", 32)),
            state_dim=int(model_config.get("state_dim", 32)),
            depth=int(model_config.get("depth", 4)),
            dropout=float(model_config.get("dropout", 0.0)),
            input_steps=input_steps,
            pred_steps=pred_steps,
            gate_limit=float(model_config.get("gate_limit", 0.25)),
            gate_bias=float(model_config.get("gate_bias", -4.0)),
            correction_scale=float(model_config.get("correction_scale", 1.0)),
        )
    if model_name in {"transformer", "transformer1d", "transformer1dbaseline"}:
        return Transformer1DBaseline(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 1)),
            d_model=int(model_config.get("d_model", 64)),
            n_heads=int(model_config.get("n_heads", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            dim_feedforward=int(model_config.get("dim_feedforward", 128)),
            dropout=float(model_config.get("dropout", 0.0)),
            input_steps=input_steps,
            pred_steps=pred_steps,
        )
    if model_name in {"transformer2d", "transformer2dbaseline"}:
        return Transformer2DBaseline(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 1)),
            d_model=int(model_config.get("d_model", 64)),
            n_heads=int(model_config.get("n_heads", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            dim_feedforward=int(model_config.get("dim_feedforward", 128)),
            dropout=float(model_config.get("dropout", 0.0)),
            input_steps=input_steps,
            pred_steps=pred_steps,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def maybe_initialize_fno_base(
    model: torch.nn.Module,
    experiment_config: dict[str, object],
) -> dict[str, object]:
    """Optionally initialize an SM-FNO residual FNO base from an FNO checkpoint."""
    checkpoint_value = experiment_config.get("fno_base_checkpoint_path")
    if checkpoint_value is None:
        return {"fno_base_initialized": False, "fno_base_checkpoint_path": None}

    checkpoint_path = Path(str(checkpoint_value))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FNO base checkpoint not found: {checkpoint_path}")
    if not hasattr(model, "base"):
        raise ValueError("fno_base_checkpoint_path requires a model with a `.base` module.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing_keys, unexpected_keys = model.base.load_state_dict(state_dict, strict=False)
    return {
        "fno_base_initialized": True,
        "fno_base_checkpoint_path": str(checkpoint_path),
        "fno_base_missing_keys": list(missing_keys),
        "fno_base_unexpected_keys": list(unexpected_keys),
    }


def main() -> None:
    """Run a minimal Heat1D training loop."""
    args = parse_args()
    experiment_config = load_yaml(args.config)
    data_config = load_yaml(Path(str(experiment_config["data_config"])))
    model_config = merge_model_config(
        experiment_config,
        load_yaml(Path(str(experiment_config["model_config"]))),
    )
    train_config = load_yaml(Path(str(experiment_config["train_config"])))

    seed = int(experiment_config.get("seed", train_config.get("seed", 42)))
    seed_everything(seed)

    input_steps = int(experiment_config.get("input_steps", 10))
    pred_steps = int(experiment_config.get("pred_steps", 1))
    rollout_train_steps = int(experiment_config.get("rollout_train_steps", 0))
    rollout_loss_weight = float(experiment_config.get("rollout_loss_weight", 0.0))
    target_steps = max(
        pred_steps,
        rollout_train_steps if rollout_loss_weight > 0.0 else pred_steps,
    )
    data_path = resolve_data_path(experiment_config, data_config)

    train_dataset, val_dataset, _ = build_trajectory_datasets(
        data_path,
        input_steps=input_steps,
        pred_steps=target_steps,
        train_ratio=float(
            experiment_config.get("train_ratio", data_config.get("train_ratio", 0.8))
        ),
        val_ratio=float(experiment_config.get("val_ratio", data_config.get("val_ratio", 0.1))),
        seed=seed,
    )

    batch_size = int(experiment_config.get("batch_size", train_config.get("batch_size", 32)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None

    device = resolve_device(str(experiment_config.get("device", train_config.get("device", "cpu"))))
    model = build_model(
        model_config,
        input_steps=input_steps,
        pred_steps=pred_steps,
        grid_size=train_dataset.grid_size,
    )
    fno_base_init = maybe_initialize_fno_base(model, experiment_config)
    parameter_counts = count_parameters(model)
    optimizer = build_optimizer(
        model.parameters(),
        name=str(experiment_config.get("optimizer", train_config.get("optimizer", "adamw"))),
        learning_rate=float(
            experiment_config.get("learning_rate", train_config.get("learning_rate", 1e-3))
        ),
        weight_decay=float(
            experiment_config.get("weight_decay", train_config.get("weight_decay", 1e-4))
        ),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        rollout_train_steps=rollout_train_steps,
        rollout_loss_weight=rollout_loss_weight,
        delta_loss_weight=float(experiment_config.get("delta_loss_weight", 0.0)),
        persistence_loss_weight=float(experiment_config.get("persistence_loss_weight", 0.0)),
        grad_clip_norm=(
            None
            if experiment_config.get("grad_clip_norm", train_config.get("grad_clip_norm")) is None
            else float(experiment_config.get("grad_clip_norm", train_config.get("grad_clip_norm")))
        ),
    )
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=int(experiment_config.get("epochs", train_config.get("epochs", 1))),
    )

    experiment_name = str(experiment_config.get("name", "experiment"))
    checkpoint_path = Path(
        str(
            experiment_config.get(
                "checkpoint_path",
                Path("results") / "checkpoints" / f"{experiment_name}.pt",
            )
        )
    )
    metrics_path = Path(
        str(
            experiment_config.get(
                "train_metrics_path",
                Path("results") / "tables" / f"{experiment_name}_train_metrics.json",
            )
        )
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_config": model_config,
            "input_steps": input_steps,
            "pred_steps": pred_steps,
            "grid_size": train_dataset.grid_size,
            "experiment_name": experiment_name,
            **fno_base_init,
        },
        checkpoint_path,
    )

    metrics = {
        "experiment": experiment_name,
        "base_experiment": str(experiment_config.get("base_experiment", experiment_name)),
        "analysis_name": str(experiment_config.get("analysis_name", "")),
        "run_type": str(
            experiment_config.get(
                "run_type",
                "repeated_seed"
                if bool(experiment_config.get("repeated_seed", False))
                else "single",
            )
        ),
        "seed": seed,
        "dataset_name": str(data_config.get("name", "unknown")),
        "model_name": str(model_config.get("name", "unknown")),
        "data_path": str(data_path),
        "checkpoint_path": str(checkpoint_path),
        "input_steps": input_steps,
        "pred_steps": pred_steps,
        "target_steps": target_steps,
        "rollout_train_steps": rollout_train_steps,
        "rollout_loss_weight": rollout_loss_weight,
        "delta_loss_weight": float(experiment_config.get("delta_loss_weight", 0.0)),
        "persistence_loss_weight": float(
            experiment_config.get("persistence_loss_weight", 0.0)
        ),
        "rollout_steps": int(experiment_config.get("rollout_steps", pred_steps)),
        **fno_base_init,
        **parameter_counts,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"[train] saved checkpoint {checkpoint_path}")
    print(f"[train] saved metrics {metrics_path}")
    print(f"[train] final train loss {history['train_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
