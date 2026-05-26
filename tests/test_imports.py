"""Import smoke tests for public package modules."""

from __future__ import annotations


def test_package_imports() -> None:
    """Verify the package and core models import successfully."""
    import sm_fno
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
        Transformer1DBaseline,
        Transformer2DBaseline,
    )

    assert sm_fno.__version__
    assert DiagonalSSM is not None
    assert FNO1D is not None
    assert FNO2D is not None
    assert MLPBaseline is not None
    assert SpectralConv1D is not None
    assert SpectralConv2D is not None
    assert SpectralMemoryFNO1D is not None
    assert SpectralMemoryFNO2D is not None
    assert SpectralMemoryFNO2DV2 is not None
    assert Transformer1DBaseline is not None
    assert Transformer2DBaseline is not None
