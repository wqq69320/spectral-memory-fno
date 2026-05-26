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
                loss = self.loss_fn(predictions, targets)
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
            loss = self.loss_fn(predictions, targets)
            batch_size = inputs.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size

        return total_loss / max(1, total_examples)

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract input and target tensors from supported batch containers."""
        if hasattr(batch, "inputs") and hasattr(batch, "targets"):
            return batch.inputs, batch.targets
        if isinstance(batch, dict):
            return batch["inputs"], batch["targets"]
        inputs, targets = batch
        return inputs, targets
