"""Run one experiment config over multiple seeds."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sm_fno.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an experiment config over repeated seeds.")
    parser.add_argument("--config", type=Path, required=True, help="Base experiment YAML config.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seed values to run.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded configs without launching training or evaluation.",
    )
    return parser.parse_args()


def build_seed_experiment_config(
    base_config: dict[str, Any],
    *,
    seed: int,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Create a per-seed experiment config with clearly named artifact paths."""
    expanded = deepcopy(base_config)
    base_name = str(expanded.get("name", config_path.stem if config_path else "experiment"))
    seed_name = f"{base_name}_seed{seed}"
    seed_output_dir = Path("outputs") / "repeated_seeds" / base_name / f"seed_{seed}"

    expanded.update(
        {
            "name": seed_name,
            "base_experiment": base_name,
            "run_type": "repeated_seed",
            "repeated_seed": True,
            "seed": int(seed),
            "output_dir": str(seed_output_dir),
            "checkpoint_path": str(Path("results") / "checkpoints" / f"{seed_name}.pt"),
            "train_metrics_path": str(
                Path("results") / "tables" / f"{seed_name}_train_metrics.json"
            ),
            "eval_metrics_path": str(
                Path("results") / "tables" / f"{seed_name}_eval_metrics.json"
            ),
            "prediction_path": str(seed_output_dir / "predictions.npz"),
        }
    )
    return expanded


def _script_env() -> dict[str, str]:
    """Return an environment that can import the local package."""
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def run_seed_config(seed_config: dict[str, Any]) -> None:
    """Run train and evaluate for one expanded seed config."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as file:
        yaml.safe_dump(seed_config, file, sort_keys=False)
        temp_config_path = Path(file.name)

    try:
        for script_name in ("train.py", "evaluate.py"):
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / script_name),
                    "--config",
                    str(temp_config_path),
                ],
                check=True,
                cwd=PROJECT_ROOT,
                env=_script_env(),
            )
    finally:
        temp_config_path.unlink(missing_ok=True)


def main() -> None:
    """Run the configured repeated-seed experiment grid."""
    args = parse_args()
    base_config = load_yaml(args.config)

    for seed in args.seeds:
        seed_config = build_seed_experiment_config(
            base_config,
            seed=seed,
            config_path=args.config,
        )
        if args.dry_run:
            print(yaml.safe_dump(seed_config, sort_keys=False).strip())
            continue

        print(
            "[run_repeated_seeds] "
            f"running {seed_config['base_experiment']} seed {seed_config['seed']}"
        )
        run_seed_config(seed_config)
        print(f"[run_repeated_seeds] wrote {seed_config['eval_metrics_path']}")


if __name__ == "__main__":
    main()
