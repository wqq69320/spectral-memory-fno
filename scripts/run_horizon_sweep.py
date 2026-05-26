"""Run CPU-friendly rollout-horizon sweeps from a YAML analysis config."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sm_fno.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run rollout-horizon cost analysis.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a horizon-sweep YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print expanded configs without launching training or evaluation.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only run evaluations. Existing per-seed checkpoints must already exist.",
    )
    parser.add_argument(
        "--reuse-existing-checkpoints",
        action="store_true",
        help="Skip training when the per-seed checkpoint path already exists.",
    )
    return parser.parse_args()


def _script_env() -> dict[str, str]:
    """Return an environment that can import the local package."""
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def _base_name(base_config: dict[str, Any], config_path: Path | None = None) -> str:
    """Resolve a stable experiment base name."""
    return str(base_config.get("name", config_path.stem if config_path else "experiment"))


def _analysis_seed_prefix(analysis_name: str) -> str:
    """Return a short seed-name prefix for analysis-expanded checkpoints."""
    for prefix in ("m9", "m10", "m11"):
        if analysis_name.startswith(prefix):
            return prefix
    return "analysis"


def build_seed_train_config(
    base_config: dict[str, Any],
    *,
    seed: int,
    analysis_name: str = "m9_cost_efficiency",
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Build the per-seed training config shared by all horizon evaluations."""
    expanded = deepcopy(base_config)
    base_name = _base_name(expanded, config_path)
    seed_name = f"{base_name}_{_analysis_seed_prefix(analysis_name)}_seed{seed}"
    output_dir = Path("outputs") / analysis_name / base_name / f"seed_{seed}"

    expanded.update(
        {
            "name": seed_name,
            "base_experiment": base_name,
            "analysis_name": analysis_name,
            "run_type": "horizon_sweep",
            "repeated_seed": True,
            "seed": int(seed),
            "output_dir": str(output_dir),
            "checkpoint_path": str(Path("results") / "checkpoints" / f"{seed_name}.pt"),
            "train_metrics_path": str(
                Path("results") / "tables" / analysis_name / f"{seed_name}_train_metrics.json"
            ),
        }
    )
    return expanded


def build_horizon_eval_config(
    base_config: dict[str, Any],
    *,
    seed: int,
    horizon: int,
    train_config: dict[str, Any],
    analysis_name: str = "m9_cost_efficiency",
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Build a horizon-specific evaluation config using a per-seed checkpoint."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    expanded = deepcopy(base_config)
    base_name = _base_name(expanded, config_path)
    run_name = f"{base_name}_h{horizon}_seed{seed}"
    output_dir = Path("outputs") / analysis_name / base_name / f"seed_{seed}" / f"horizon_{horizon}"

    expanded.update(
        {
            "name": run_name,
            "base_experiment": base_name,
            "analysis_name": analysis_name,
            "run_type": "horizon_sweep",
            "repeated_seed": True,
            "seed": int(seed),
            "rollout_steps": int(horizon),
            "output_dir": str(output_dir),
            "checkpoint_path": str(train_config["checkpoint_path"]),
            "eval_metrics_path": str(
                Path("results") / "tables" / analysis_name / f"{run_name}_eval_metrics.json"
            ),
            "prediction_path": str(output_dir / "predictions.npz"),
        }
    )
    return expanded


def write_expanded_config(config: dict[str, Any], config_dir: Path) -> Path:
    """Write an expanded YAML config for reproducibility."""
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{config['name']}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_script(script_name: str, config_path: Path) -> None:
    """Run one repository script with an expanded YAML config."""
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / script_name),
            "--config",
            str(config_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=_script_env(),
    )


def run_horizon_sweep(
    analysis_config: dict[str, Any],
    *,
    dry_run: bool = False,
    skip_train: bool = False,
    reuse_existing_checkpoints: bool = False,
) -> list[dict[str, str]]:
    """Run the configured horizon sweep and return generated config paths."""
    analysis_name = str(analysis_config.get("name", "m9_cost_efficiency"))
    experiment_paths = [Path(str(path)) for path in analysis_config["experiments"]]
    seeds = [int(seed) for seed in analysis_config.get("seeds", [0])]
    horizons = [int(horizon) for horizon in analysis_config.get("horizons", [5, 10, 20])]
    config_dir = Path(str(analysis_config.get("expanded_config_dir", "outputs"))) / (
        f"{analysis_name}_expanded_configs"
    )

    written_configs: list[dict[str, str]] = []
    for experiment_path in experiment_paths:
        base_config = load_yaml(experiment_path)
        for seed in seeds:
            train_config = build_seed_train_config(
                base_config,
                seed=seed,
                analysis_name=analysis_name,
                config_path=experiment_path,
            )
            train_config_path = write_expanded_config(train_config, config_dir)
            written_configs.append({"kind": "train", "path": str(train_config_path)})

            checkpoint_path = PROJECT_ROOT / str(train_config["checkpoint_path"])
            should_train = not skip_train and not (
                reuse_existing_checkpoints and checkpoint_path.exists()
            )
            if dry_run:
                print(yaml.safe_dump(train_config, sort_keys=False).strip())
            elif should_train:
                print(
                    "[run_horizon_sweep] "
                    f"training {train_config['base_experiment']} seed {seed}"
                )
                run_script("train.py", train_config_path)

            for horizon in horizons:
                eval_config = build_horizon_eval_config(
                    base_config,
                    seed=seed,
                    horizon=horizon,
                    train_config=train_config,
                    analysis_name=analysis_name,
                    config_path=experiment_path,
                )
                eval_config_path = write_expanded_config(eval_config, config_dir)
                written_configs.append({"kind": "eval", "path": str(eval_config_path)})
                if dry_run:
                    print(yaml.safe_dump(eval_config, sort_keys=False).strip())
                    continue

                print(
                    "[run_horizon_sweep] "
                    f"evaluating {eval_config['base_experiment']} seed {seed} horizon {horizon}"
                )
                run_script("evaluate.py", eval_config_path)

    return written_configs


def main() -> None:
    """Run the configured horizon sweep."""
    args = parse_args()
    analysis_config = load_yaml(args.config)
    written_configs = run_horizon_sweep(
        analysis_config,
        dry_run=args.dry_run,
        skip_train=args.skip_train,
        reuse_existing_checkpoints=args.reuse_existing_checkpoints,
    )
    print(f"[run_horizon_sweep] wrote {len(written_configs)} expanded configs")


if __name__ == "__main__":
    main()
