"""Run a minimal smoke test for the scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sm_fno.models.sm_fno1d import SpectralMemoryFNO1D
from sm_fno.utils.config import load_yaml
from sm_fno.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run scaffold smoke test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/smoke_test.yaml"),
        help="Path to the smoke-test experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    """Instantiate SM-FNO and verify a forward pass shape."""
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config.get("seed", 42)))

    batch_size = int(config.get("batch_size", 2))
    grid_size = int(config.get("grid_size", 32))

    model = SpectralMemoryFNO1D(in_channels=1, out_channels=1, modes=8, width=16)
    inputs = torch.randn(batch_size, grid_size, 1)
    outputs = model(inputs)

    if outputs.shape != inputs.shape:
        raise RuntimeError(f"Expected output shape {inputs.shape}, got {outputs.shape}.")

    print("[smoke_test] OK: forward pass shape matches input shape.")


if __name__ == "__main__":
    main()
