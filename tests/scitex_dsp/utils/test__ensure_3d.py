"""Tests for scitex_dsp.utils._ensure_3d."""

import numpy as np

from scitex_dsp.utils._ensure_3d import ensure_3d


def test_ensure_3d_from_1d_out_ndim_equals_n_3():
    # Arrange
    x = np.zeros(8)
    # Act
    out = ensure_3d(x)
    # Act
    # Assert
    assert out.ndim == 3


def test_ensure_3d_from_1d_out_shape_equals_n_1_1_8():
    # Arrange
    x = np.zeros(8)
    # Act
    out = ensure_3d(x)
    # Act
    # Assert
    assert out.shape == (1, 1, 8)




def test_ensure_3d_from_2d_out_ndim_equals_n_3():
    # Arrange
    x = np.zeros((4, 8))
    # Act
    out = ensure_3d(x)
    # Act
    # Assert
    assert out.ndim == 3


def test_ensure_3d_from_2d_out_shape_equals_n_4_1_8():
    # Arrange
    x = np.zeros((4, 8))
    # Act
    out = ensure_3d(x)
    # Act
    # Assert
    assert out.shape == (4, 1, 8)




def test_ensure_3d_passthrough_3d_out_ndim_equals_n_3():
    # Arrange
    x = np.zeros((2, 3, 8))
    # Act
    out = ensure_3d(x)
    # Act
    # Assert
    assert out.ndim == 3


def test_ensure_3d_passthrough_3d_out_shape_equals_n_2_3_8():
    # Arrange
    x = np.zeros((2, 3, 8))
    # Act
    out = ensure_3d(x)
    # Act
    # Assert
    assert out.shape == (2, 3, 8)




if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
