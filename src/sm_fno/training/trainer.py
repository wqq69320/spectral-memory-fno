"""Training loop scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from sm_fno.training.losses import mse_loss


@dataclass
class Trainer:
    """Minimal trainer for small PDE forecasting experiments."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss
    grad_clip_norm: float | None = None
    rollout_train_steps: int = 0
    rollout_loss_weight: float = 0.0
    delta_loss_weight: float = 0.0
    persistence_loss_weight: float = 0.0

    def fit(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
        *,
        epochs: int = 1,
    ) -> dict[str, list[float]]:
        """Train the model and return loss history."""
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        self.model.to(self.device)
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for _ in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_examples = 0

            for batch in train_loader:
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                predictions = self.model(inputs)
                loss = self._compute_loss(inputs, targets, predictions)
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                batch_size = inputs.shape[0]
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size

            history["train_loss"].append(total_loss / max(1, total_examples))
            if val_loader is not None:
                history["val_loss"].append(self.evaluate_loss(val_loader))

        return history

    @torch.no_grad()
    def evaluate_loss(self, data_loader: Any) -> float:
        """Compute average loss on a data loader."""
        self.model.eval()
        total_loss = 0.0
        total_examples = 0

        for batch in data_loader:
            inputs, targets = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            predictions = self.model(inputs)
            loss = self._compute_loss(inputs, targets, predictions)
            batch_size = inputs.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size

        return total_loss / max(1, total_examples)

    def _compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one-step loss plus optional dynamic and rollout terms."""
        one_step_targets = targets[:, : predictions.shape[1]]
        loss = self._prediction_loss(inputs, predictions, one_step_targets)
        if self.rollout_loss_weight <= 0.0 or self.rollout_train_steps <= 0:
            return loss
        if self.rollout_train_steps > targets.shape[1]:
            raise ValueError(
                "rollout_train_steps exceeds available target steps: "
                f"{self.rollout_train_steps} > {targets.shape[1]}."
            )
        rollout_predictions = self._differentiable_rollout(inputs, self.rollout_train_steps)
        rollout_targets = targets[:, : self.rollout_train_steps]
        rollout_loss = self._prediction_loss(inputs, rollout_predictions, rollout_targets)
        return loss + self.rollout_loss_weight * rollout_loss

    def _prediction_loss(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE plus optional delta and persistence-aware penalties."""
        loss = self.loss_fn(predictions, targets)
        if self.delta_loss_weight > 0.0:
            reference = inputs[:, -1:]
            predicted_delta = predictions - reference
            target_delta = targets - reference
            loss = loss + self.delta_loss_weight * self.loss_fn(predicted_delta, target_delta)
        if self.persistence_loss_weight > 0.0:
            reference = inputs[:, -1:]
            persistence = reference.repeat(1, targets.shape[1], *([1] * (targets.ndim - 2)))
            model_error = self.loss_fn(predictions, targets)
            persistence_error = self.loss_fn(persistence, targets).detach()
            persistence_advantage = torch.relu(model_error - persistence_error)
            loss = loss + self.persistence_loss_weight * persistence_advantage
        return loss

    def _differentiable_rollout(self, initial_state: torch.Tensor, steps: int) -> torch.Tensor:
        """Roll out the model while preserving gradients for training."""
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        current_window = initial_state
        predictions: list[torch.Tensor] = []
        while len(predictions) < steps:
            next_predictions = self.model(current_window)
            if next_predictions.ndim != current_window.ndim:
                raise ValueError(
                    "rollout-aware training expects windowed model outputs with shape "
                    "(batch, pred_time, *spatial, channels)."
                )
            for time_index in range(next_predictions.shape[1]):
                next_step = next_predictions[:, time_index : time_index + 1]
                predictions.append(next_step[:, 0])
                current_window = torch.cat([current_window[:, 1:], next_step], dim=1)
                if len(predictions) == steps:
                    break
        return torch.stack(predictions, dim=1)

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract input and target tensors from supported batch containers."""
        if hasattr(batch, "inputs") and hasattr(batch, "targets"):
            return batch.inputs, batch.targets
        if isinstance(batch, dict):
            return batch["inputs"], batch["targets"]
        inputs, targets = batch
        return inputs, targets
