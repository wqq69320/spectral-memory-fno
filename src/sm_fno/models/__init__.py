"""Model definitions and builders."""

from __future__ import annotations

from sm_fno.models.base import ForecastModel
from sm_fno.models.fno1d import FNO1D, SpectralConv1D, SpectralConv1d
from sm_fno.models.fno2d import FNO2D, SpectralConv2D
from sm_fno.models.mlp import MLPBaseline
from sm_fno.models.sm_fno1d import SpectralMemoryFNO1D
from sm_fno.models.sm_fno2d import SpectralMemoryFNO2D, SpectralMemoryFNO2DV2
from sm_fno.models.ssm import DiagonalSSM, StableGatedDiagonalSSM
from sm_fno.models.transformer1d import Transformer1DBaseline
from sm_fno.models.transformer2d import Transformer2DBaseline

__all__ = [
    "DiagonalSSM",
    "FNO1D",
    "FNO2D",
    "ForecastModel",
    "MLPBaseline",
    "SpectralConv1D",
    "SpectralConv1d",
    "SpectralConv2D",
    "SpectralMemoryFNO1D",
    "SpectralMemoryFNO2D",
    "SpectralMemoryFNO2DV2",
    "StableGatedDiagonalSSM",
    "Transformer1DBaseline",
    "Transformer2DBaseline",
]
