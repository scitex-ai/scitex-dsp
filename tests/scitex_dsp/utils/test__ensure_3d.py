"""Tests for scitex_dsp.utils._ensure_3d."""

import numpy as np

from scitex_dsp.utils._ensure_3d import ensure_3d


def test_ensure_3d_from_1d():
    x = np.zeros(8)
    out = ensure_3d(x)
    assert out.ndim == 3
    assert out.shape == (1, 1, 8)


def test_ensure_3d_from_2d():
    x = np.zeros((4, 8))
    out = ensure_3d(x)
    assert out.ndim == 3
    assert out.shape == (4, 1, 8)


def test_ensure_3d_passthrough_3d():
    x = np.zeros((2, 3, 8))
    out = ensure_3d(x)
    assert out.ndim == 3
    assert out.shape == (2, 3, 8)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
