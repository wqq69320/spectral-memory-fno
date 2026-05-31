"""Shape tests for Heat1D research MVP components."""

from __future__ import annotations

import torch

from sm_fno.evaluation.rollout import autoregressive_rollout
from sm_fno.models import (
    DiagonalSSM,
    FNO1D,
    FNO2D,
    MLPBaseline,
    SpectralConv1D,
    SpectralConv2D,
    SpectralMemoryFNO1D,
    SpectralMemoryFNO2D,
    SpectralMemoryFNO2DV2,
    SpectralMemoryFNO2DV3,
    StableGatedDiagonalSSM,
    Transformer1DBaseline,
    Transformer2DBaseline,
)
from sm_fno.training import Trainer


def test_legacy_model_output_shapes_match_input_grid_shape() -> None:
    """Legacy one-step models should preserve batch and grid dimensions."""
    inputs = torch.randn(2, 32, 1)
    models = [
        MLPBaseline(in_channels=1, out_channels=1, hidden_dim=8, num_layers=2),
        FNO1D(in_channels=1, out_channels=1, modes=8, width=16, depth=2),
        Transformer1DBaseline(
            in_channels=1,
            out_channels=1,
            d_model=16,
            n_heads=4,
            num_layers=1,
            dim_feedforward=32,
        ),
        SpectralMemoryFNO1D(in_channels=1, out_channels=1, modes=8, width=16, depth=2),
    ]

    for model in models:
        outputs = model(inputs)
        assert outputs.shape == inputs.shape


def test_spectral_conv1d_shape() -> None:
    """SpectralConv1D should preserve batch/grid and emit configured channels."""
    layer = SpectralConv1D(in_channels=3, out_channels=5, modes=8)
    inputs = torch.randn(2, 3, 32)
    outputs = layer(inputs)
    assert outputs.shape == (2, 5, 32)


def test_spectral_conv2d_shape() -> None:
    """SpectralConv2D should preserve batch/spatial shape and emit channels."""
    layer = SpectralConv2D(in_channels=3, out_channels=5, modes=4)
    inputs = torch.randn(2, 3, 16, 16)
    outputs = layer(inputs)
    assert outputs.shape == (2, 5, 16, 16)


def test_spectral_conv2d_small_grid_modes_do_not_overlap() -> None:
    """Top and bottom y-frequency writes should be disjoint on small grids."""
    layer = SpectralConv2D(in_channels=1, out_channels=1, modes=8)

    assert layer.retained_mode_counts(height=8, width_rfft=5) == (4, 4, 5)
    assert layer.retained_mode_counts(height=15, width_rfft=8) == (8, 7, 8)

    inputs = torch.randn(2, 1, 8, 8)
    outputs = layer(inputs)
    outputs.square().mean().backward()

    assert layer.weight_top.grad is not None
    assert layer.weight_bottom.grad is not None
    assert layer.weight_top.grad[:, :, :4, :5].abs().sum() > 0
    assert layer.weight_bottom.grad[:, :, :4, :5].abs().sum() > 0


def test_fno1d_window_forward_shape_and_backward() -> None:
    """FNO1D should map input windows to prediction windows with gradients."""
    model = FNO1D(
        in_channels=1,
        out_channels=1,
        modes=8,
        width=16,
        depth=2,
        input_steps=4,
        pred_steps=2,
    )
    inputs = torch.randn(2, 4, 32, 1)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 32, 1)
    outputs.square().mean().backward()
    assert model.lift.weight.grad is not None


def test_fno2d_window_forward_shape_and_backward() -> None:
    """FNO2D should map 2D input windows to 2D prediction windows."""
    model = FNO2D(
        in_channels=1,
        out_channels=1,
        modes=4,
        width=8,
        depth=1,
        input_steps=4,
        pred_steps=2,
    )
    inputs = torch.randn(2, 4, 12, 12, 1)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 12, 12, 1)
    outputs.square().mean().backward()
    assert model.lift.weight.grad is not None


def test_diagonal_ssm_forward_shape() -> None:
    """DiagonalSSM should preserve batch/time and emit configured width."""
    model = DiagonalSSM(input_dim=6, state_dim=8, output_dim=5)
    inputs = torch.randn(2, 4, 6)
    outputs, state = model(inputs)
    assert outputs.shape == (2, 4, 5)
    assert state.shape == (2, 8)


def test_stable_gated_diagonal_ssm_transition_range_and_shape() -> None:
    """Stable gated SSM should keep transition values inside (0, 1)."""
    model = StableGatedDiagonalSSM(input_dim=6, state_dim=8, output_dim=5)
    inputs = torch.randn(2, 4, 6)
    outputs, state = model(inputs)
    transition = model.stable_transition()

    assert outputs.shape == (2, 4, 5)
    assert state.shape == (2, 8)
    assert torch.all(transition > 0.0)
    assert torch.all(transition < 1.0)


def test_sm_fno1d_window_forward_shape_and_backward() -> None:
    """SM-FNO should map input windows to prediction windows with gradients."""
    model = SpectralMemoryFNO1D(
        in_channels=1,
        out_channels=1,
        modes=8,
        width=16,
        state_dim=16,
        depth=2,
        input_steps=4,
        pred_steps=2,
    )
    inputs = torch.randn(2, 4, 32, 1)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 32, 1)
    outputs.square().mean().backward()
    assert model.lift.weight.grad is not None


def test_sm_fno2d_window_forward_shape_and_backward() -> None:
    """SM-FNO2D should map 2D input windows to 2D prediction windows."""
    model = SpectralMemoryFNO2D(
        in_channels=1,
        out_channels=1,
        modes=4,
        width=8,
        state_dim=8,
        depth=1,
        input_steps=4,
        pred_steps=2,
    )
    inputs = torch.randn(2, 4, 12, 12, 1)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 12, 12, 1)
    outputs.square().mean().backward()
    assert model.lift.weight.grad is not None


def test_sm_fno2d_v2_window_forward_shape_and_backward() -> None:
    """SM-FNO2D v2 should map 2D input windows to 2D prediction windows."""
    model = SpectralMemoryFNO2DV2(
        in_channels=1,
        out_channels=1,
        modes=4,
        width=8,
        state_dim=8,
        depth=1,
        input_steps=4,
        pred_steps=2,
    )
    inputs = torch.randn(2, 4, 12, 12, 1)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 12, 12, 1)
    outputs.square().mean().backward()
    assert model.lift.weight.grad is not None
    assert model.memory.log_decay.grad is not None


def test_sm_fno2d_v3_window_forward_shape_and_backward() -> None:
    """SM-FNO2D v3 should map 2D input windows to 2D prediction windows."""
    model = SpectralMemoryFNO2DV3(
        in_channels=1,
        out_channels=1,
        modes=4,
        width=8,
        state_dim=8,
        depth=1,
        input_steps=4,
        pred_steps=2,
        gate_limit=0.25,
    )
    inputs = torch.randn(2, 4, 12, 12, 1)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 12, 12, 1)
    outputs.square().mean().backward()
    assert model.base.lift.weight.grad is not None
    assert model.memory.log_decay.grad is not None
    assert model.gate_project.bias.grad is not None


def test_transformer1d_window_forward_shape_and_backward() -> None:
    """Transformer1D should map input windows to prediction windows with gradients."""
    model = Transformer1DBaseline(
        in_channels=1,
        out_channels=1,
        d_model=16,
        n_heads=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        input_steps=4,
        pred_steps=2,
    )
    inputs = torch.randn(2, 4, 16, 1)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 16, 1)
    outputs.square().mean().backward()
    assert model.lift.weight.grad is not None


def test_transformer2d_window_forward_shape_and_backward() -> None:
    """Transformer2D should map 2D input windows to 2D prediction windows."""
    model = Transformer2DBaseline(
        in_channels=1,
        out_channels=1,
        d_model=16,
        n_heads=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        input_steps=4,
        pred_steps=2,
    )
    inputs = torch.randn(2, 4, 8, 8, 1)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 8, 8, 1)
    outputs.square().mean().backward()
    assert model.lift.weight.grad is not None


def test_autoregressive_rollout_shape() -> None:
    """Rollout utility should stack generated predictions along time."""
    model = SpectralMemoryFNO1D(
        in_channels=1,
        out_channels=1,
        modes=8,
        width=16,
        state_dim=16,
        depth=2,
        input_steps=3,
        pred_steps=1,
    )
    initial_state = torch.randn(2, 3, 32, 1)
    rollout = autoregressive_rollout(model, initial_state, steps=4)
    assert rollout.shape == (2, 4, 32, 1)


def test_autoregressive_rollout_2d_shape() -> None:
    """Rollout utility should preserve 2D spatial dimensions."""
    model = FNO2D(
        in_channels=1,
        out_channels=1,
        modes=4,
        width=8,
        depth=1,
        input_steps=3,
        pred_steps=1,
    )
    initial_state = torch.randn(2, 3, 8, 8, 1)
    rollout = autoregressive_rollout(model, initial_state, steps=4)
    assert rollout.shape == (2, 4, 8, 8, 1)


def test_rollout_aware_trainer_accepts_longer_targets() -> None:
    """Rollout-aware training should use multi-step targets with a one-step model."""

    class LastStepScale(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.scale * inputs[:, -1:]

    model = LastStepScale()
    inputs = torch.randn(2, 3, 8, 1)
    targets = torch.randn(2, 4, 8, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        rollout_train_steps=4,
        rollout_loss_weight=0.2,
    )

    history = trainer.fit([(inputs, targets)], epochs=1)

    assert len(history["train_loss"]) == 1
    assert model.scale.grad is not None


def test_mlp_flattened_heat1d_shape() -> None:
    """Flattened MLP mode should map input windows to prediction windows."""
    model = MLPBaseline(
        in_channels=1,
        out_channels=1,
        hidden_dim=16,
        num_layers=2,
        input_steps=4,
        pred_steps=2,
        grid_size=16,
    )
    inputs = torch.randn(3, 4, 16, 1)
    outputs = model(inputs)
    assert outputs.shape == (3, 2, 16, 1)
