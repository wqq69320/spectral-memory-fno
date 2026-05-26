"""Base model interfaces for PDE forecasting."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ForecastModel(nn.Module, ABC):
    """Abstract base class for one-step PDE forecasters."""

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the next state from input states."""
        raise NotImplementedError
