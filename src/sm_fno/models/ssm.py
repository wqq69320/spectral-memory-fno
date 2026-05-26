"""State space memory modules."""

from __future__ import annotations

import torch
from torch import nn


class DiagonalSSM(nn.Module):
    """Simple diagonal state space model.

    The hidden state follows the documented recurrence

    ``h_t = A h_{t-1} + B x_t``

    where ``A`` is represented as a learned diagonal transition and ``B`` is a
    learned linear input map. The transition is constrained to ``(-1, 1)`` with
    ``tanh`` to keep the small CPU smoke experiments numerically stable.
    """

    def __init__(self, input_dim: int, state_dim: int, output_dim: int | None = None) -> None:
        """Initialize the diagonal state space recurrence."""
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if state_dim < 1:
            raise ValueError("state_dim must be positive.")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.state_transition = nn.Parameter(torch.zeros(state_dim))
        self.input_to_state = nn.Linear(input_dim, state_dim)
        self.state_to_output = nn.Linear(state_dim, self.output_dim)

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        """Create a zero initial state."""
        return torch.zeros(batch_size, self.state_dim, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrence over an input sequence.

        Args:
            inputs: Tensor with shape ``(batch, time, input_dim)``.
            state: Optional tensor with shape ``(batch, state_dim)``.

        Returns:
            A tuple ``(outputs, next_state)`` where outputs has shape
            ``(batch, time, output_dim)``.
        """
        if inputs.ndim != 3:
            raise ValueError("DiagonalSSM expects inputs with shape (batch, time, input_dim).")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {inputs.shape[-1]}.")
        if state is None:
            state = self.initial_state(inputs.shape[0], device=inputs.device)
        if state.shape != (inputs.shape[0], self.state_dim):
            raise ValueError(
                "state must have shape "
                f"({inputs.shape[0]}, {self.state_dim}), got {tuple(state.shape)}."
            )

        transition = torch.tanh(self.state_transition).to(device=inputs.device)
        outputs: list[torch.Tensor] = []
        next_state = state
        for time_index in range(inputs.shape[1]):
            next_state = transition * next_state + self.input_to_state(inputs[:, time_index])
            outputs.append(self.state_to_output(next_state))

        return torch.stack(outputs, dim=1), next_state


class StableGatedDiagonalSSM(nn.Module):
    """Stable diagonal SSM with gated residual temporal outputs.

    The transition is parameterized as ``exp(-softplus(log_decay))``, keeping
    each diagonal coefficient in ``(0, 1)``. Each recurrent update blends the
    previous state with a bounded input proposal, and each output is mixed with
    an input residual through a learned gate.
    """

    def __init__(self, input_dim: int, state_dim: int, output_dim: int | None = None) -> None:
        """Initialize the stable gated diagonal SSM."""
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if state_dim < 1:
            raise ValueError("state_dim must be positive.")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.log_decay = nn.Parameter(torch.zeros(state_dim))
        self.input_to_state = nn.Linear(input_dim, state_dim)
        self.state_to_output = nn.Linear(state_dim, self.output_dim)
        self.input_residual = nn.Linear(input_dim, self.output_dim)
        self.output_gate = nn.Linear(input_dim, self.output_dim)

    def stable_transition(self) -> torch.Tensor:
        """Return diagonal transition values constrained to ``(0, 1)``."""
        return torch.exp(-torch.nn.functional.softplus(self.log_decay))

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        """Create a zero initial state."""
        return torch.zeros(batch_size, self.state_dim, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the stable gated recurrence over an input sequence."""
        if inputs.ndim != 3:
            raise ValueError(
                "StableGatedDiagonalSSM expects inputs with shape (batch, time, input_dim)."
            )
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {inputs.shape[-1]}.")
        if state is None:
            state = self.initial_state(inputs.shape[0], device=inputs.device)
        if state.shape != (inputs.shape[0], self.state_dim):
            raise ValueError(
                "state must have shape "
                f"({inputs.shape[0]}, {self.state_dim}), got {tuple(state.shape)}."
            )

        transition = self.stable_transition().to(device=inputs.device)
        outputs: list[torch.Tensor] = []
        next_state = state
        for time_index in range(inputs.shape[1]):
            current_input = inputs[:, time_index]
            proposal = torch.tanh(self.input_to_state(current_input))
            next_state = transition * next_state + (1.0 - transition) * proposal
            memory_output = self.state_to_output(next_state)
            residual_output = self.input_residual(current_input)
            gate = torch.sigmoid(self.output_gate(current_input))
            outputs.append(gate * memory_output + (1.0 - gate) * residual_output)

        return torch.stack(outputs, dim=1), next_state


class SimpleSSMCell(nn.Module):
    """Compatibility wrapper around ``DiagonalSSM`` for one-step callers."""

    def __init__(self, input_dim: int, state_dim: int) -> None:
        """Initialize the wrapped state space model."""
        super().__init__()
        self.ssm = DiagonalSSM(input_dim=input_dim, state_dim=state_dim, output_dim=input_dim)

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        """Create a zero initial state."""
        return self.ssm.initial_state(batch_size, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update memory from one input vector."""
        if inputs.ndim != 2:
            raise ValueError("SimpleSSMCell expects inputs with shape (batch, input_dim).")
        outputs, next_state = self.ssm(inputs.unsqueeze(1), state)
        return outputs[:, 0], next_state
