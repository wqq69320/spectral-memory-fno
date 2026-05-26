"""Generate PDE datasets from a plain YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sm_fno.data.burgers1d import generate_burgers1d
from sm_fno.data.heat1d import generate_heat1d
from sm_fno.data.navier_stokes2d import generate_navier_stokes2d
from sm_fno.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate PDE data.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a data YAML config.")
    return parser.parse_args()


def resolve_output_path(config: dict[str, object]) -> Path:
    """Resolve the output ``.npz`` path from a data config."""
    if "output_path" in config:
        return Path(str(config["output_path"]))
    dataset_name = str(config.get("name", "dataset"))
    output_dir = Path(str(config.get("output_dir", Path("data") / "processed" / dataset_name)))
    return output_dir / f"{dataset_name}.npz"


def main() -> None:
    """Generate the configured dataset and save it to disk."""
    args = parse_args()
    config = load_yaml(args.config)
    dataset_name = config.get("name", "unknown")

    if dataset_name not in {"heat1d", "burgers1d", "navier_stokes2d"}:
        raise ValueError(f"Unsupported dataset for generation: {dataset_name}")

    common_kwargs = {
        "num_samples": int(config.get("num_samples", 128)),
        "grid_size": int(config.get("grid_size", 64)),
        "time_steps": int(config.get("time_steps", 50)),
        "dt": float(config.get("dt", 0.001)),
        "domain_length": float(config.get("domain_length", 1.0)),
        "seed": int(config.get("seed", 42)),
    }
    if dataset_name == "heat1d":
        trajectories = generate_heat1d(
            **common_kwargs,
            alpha=float(config.get("alpha", 0.01)),
        )
        pde_metadata = {"alpha": float(config.get("alpha", 0.01))}
    elif dataset_name == "burgers1d":
        trajectories = generate_burgers1d(
            **common_kwargs,
            viscosity=float(config.get("viscosity", 0.01)),
        )
        pde_metadata = {"viscosity": float(config.get("viscosity", 0.01))}
    else:
        trajectories = generate_navier_stokes2d(
            **common_kwargs,
            viscosity=float(config.get("viscosity", 0.001)),
            cfl_safety=float(config.get("cfl_safety", 0.95)),
        )
        pde_metadata = {
            "viscosity": float(config.get("viscosity", 0.001)),
            "cfl_safety": float(config.get("cfl_safety", 0.95)),
        }

    metadata = {
        "name": dataset_name,
        "num_samples": int(config.get("num_samples", trajectories.shape[0])),
        "grid_size": int(config.get("grid_size", trajectories.shape[2])),
        "time_steps": int(config.get("time_steps", trajectories.shape[1])),
        "channels": int(trajectories.shape[-1]),
        "dt": float(config.get("dt", 0.001)),
        "domain_length": float(config.get("domain_length", 1.0)),
        "seed": int(config.get("seed", 42)),
        **pde_metadata,
    }
    if trajectories.ndim == 5:
        metadata.update(
            {
                "height": int(trajectories.shape[2]),
                "width": int(trajectories.shape[3]),
            }
        )

    output_path = resolve_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        trajectories=trajectories.numpy(),
        metadata=json.dumps(metadata),
        **metadata,
    )

    print(f"[generate_data] saved {output_path}")
    print(f"[generate_data] shape {tuple(trajectories.shape)}")


if __name__ == "__main__":
    main()
