"""Plot experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sm_fno.utils.config import load_yaml
from sm_fno.visualization import plot_heat1d_prediction, plot_rollout_error, plot_training_loss


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot SM-FNO experiment results.")
    parser.add_argument("--config", type=Path, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result tables and figure outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Create available figures for an experiment."""
    args = parse_args()
    if args.config is None:
        print("[plot_results] provide --config to plot a specific experiment.")
        return

    experiment_config = load_yaml(args.config)
    experiment_name = str(experiment_config.get("name", "experiment"))
    figures_dir = Path(str(experiment_config.get("figures_dir", args.results_dir / "figures")))
    train_metrics_path = Path(
        str(
            experiment_config.get(
                "train_metrics_path",
                args.results_dir / "tables" / f"{experiment_name}_train_metrics.json",
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
    eval_metrics_path = Path(
        str(
            experiment_config.get(
                "eval_metrics_path",
                args.results_dir / "tables" / f"{experiment_name}_eval_metrics.json",
            )
        )
    )

    if train_metrics_path.exists():
        with train_metrics_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
        loss_path = figures_dir / f"{experiment_name}_training_loss.png"
        plot_training_loss(history, loss_path)
        print(f"[plot_results] saved {loss_path}")
    else:
        print(f"[plot_results] missing training metrics: {train_metrics_path}")

    if prediction_path.exists():
        with np.load(prediction_path) as archive:
            prediction_figure_path = figures_dir / f"{experiment_name}_prediction.png"
            plot_heat1d_prediction(
                target=archive["targets"],
                prediction=archive["predictions"],
                input_state=archive["inputs"],
                output_path=prediction_figure_path,
            )
        print(f"[plot_results] saved {prediction_figure_path}")
    else:
        print(f"[plot_results] missing prediction file: {prediction_path}")

    if eval_metrics_path.exists():
        with eval_metrics_path.open("r", encoding="utf-8") as file:
            eval_metrics = json.load(file)
        rollout_errors = eval_metrics.get("rollout_relative_l2_per_timestep", [])
        if rollout_errors:
            rollout_path = figures_dir / f"{experiment_name}_rollout_relative_l2.png"
            plot_rollout_error(rollout_errors, rollout_path)
            print(f"[plot_results] saved {rollout_path}")
    else:
        print(f"[plot_results] missing evaluation metrics: {eval_metrics_path}")


if __name__ == "__main__":
    main()
