import pytest

torch = pytest.importorskip("torch")
import numpy as np

import scitex


class TestZ:
    """Test z-score normalization function."""

    def test_import_hasattr_scitex_dsp_norm_minmax(self):
        """Test function can be imported."""
        # Arrange
        # Act
        # Assert
        assert hasattr(scitex.dsp.norm, "z")

    def test_basic_1d_np_abs_np_mean_result_1e_05(self):
        # Arrange
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Act
        result = scitex.dsp.norm.z(signal)
        # Act
        # Assert
        assert np.abs(np.mean(result)) < 1e-5

    def test_basic_1d_np_abs_np_std_result_1_0_0_15(self):
        # Arrange
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Act
        result = scitex.dsp.norm.z(signal)
        # Act
        # Assert
        assert np.abs(np.std(result) - 1.0) < 0.15


    def test_basic_2d_np_all_np_abs_row_means_1e_05(self):
        # Arrange
        signal = np.random.randn(10, 100)
        result = scitex.dsp.norm.z(signal, dim=-1)
        # Check each row has zero mean (float32 tolerance)
        row_means = np.mean(result, axis=-1)
        # Act
        row_stds = np.std(result, axis=-1)
        # Act
        # Assert
        assert np.all(np.abs(row_means) < 1e-5)

    def test_basic_2d_np_all_np_abs_row_stds_1_0_0_02(self):
        # Arrange
        signal = np.random.randn(10, 100)
        result = scitex.dsp.norm.z(signal, dim=-1)
        # Check each row has zero mean (float32 tolerance)
        row_means = np.mean(result, axis=-1)
        # Act
        row_stds = np.std(result, axis=-1)
        # Act
        # Assert
        assert np.all(np.abs(row_stds - 1.0) < 0.02)


    def test_basic_3d_result_shape_equals_signal_shape(self):
        """Test z-score normalization on 3D signal preserves shape."""
        # Arrange
        signal = np.random.randn(5, 10, 100)
        # Act
        result = scitex.dsp.norm.z(signal, dim=-1)
        # Assert
        assert result.shape == signal.shape

    def test_basic_3d_all_rows_zero_mean(self):
        """Test z-score normalization on 3D signal: every row has zero mean."""
        # Arrange
        signal = np.random.randn(5, 10, 100)
        # Act
        result = scitex.dsp.norm.z(signal, dim=-1)
        # Assert
        assert all(np.abs(np.mean(row)) < 1e-5 for row in result.reshape(-1, 100))

    def test_basic_3d_all_rows_unit_std(self):
        """Test z-score normalization on 3D signal: every row has ~unit std."""
        # Arrange
        signal = np.random.randn(5, 10, 100)
        # Act
        result = scitex.dsp.norm.z(signal, dim=-1)
        # Assert
        assert all(np.abs(np.std(row) - 1.0) < 0.02 for row in result.reshape(-1, 100))

    def test_different_dimensions_np_all_np_abs_np_mean_result0_axis_0_1e_05(self):
        # Arrange
        signal = np.random.randn(4, 5, 6)
        # Test along dim=0 (float32 tolerance)
        # Act
        result0 = scitex.dsp.norm.z(signal, dim=0)
        # Act
        # Assert
        assert np.all(np.abs(np.mean(result0, axis=0)) < 1e-5)

    def test_different_dimensions_dim1_zero_mean(self):
        """z-score along dim=1 yields zero mean along that axis."""
        # Arrange
        signal = np.random.randn(4, 5, 6)
        # Act
        result1 = scitex.dsp.norm.z(signal, dim=1)
        # Assert
        assert np.all(np.abs(np.mean(result1, axis=1)) < 1e-5)

    def test_different_dimensions_dim2_zero_mean(self):
        """z-score along dim=2 yields zero mean along that axis."""
        # Arrange
        signal = np.random.randn(4, 5, 6)
        # Act
        result2 = scitex.dsp.norm.z(signal, dim=2)
        # Assert
        assert np.all(np.abs(np.mean(result2, axis=2)) < 1e-5)



    def test_constant_signal_np_all_np_isnan_result_or_np_all_np_isinf_result(self):
        """Test z-score normalization on constant signal."""
        # Arrange
        signal = np.ones((5, 10))
        # Act
        result = scitex.dsp.norm.z(signal)
        # Division by zero results in NaN values (std=0 for constant signal)
        # Assert
        assert np.all(np.isnan(result)) or np.all(np.isinf(result))

    def test_torch_input_result_is_torch_tensor(self):
        # Arrange
        signal = torch.randn(10, 50)
        # Act
        result = scitex.dsp.norm.z(signal)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_torch_abs_torch_mean_result_dim_1_max_1e_05(self):
        # Arrange
        signal = torch.randn(10, 50)
        # Act
        result = scitex.dsp.norm.z(signal)
        # Act
        # Assert
        assert torch.abs(torch.mean(result, dim=-1)).max() < 1e-5

    def test_torch_input_torch_abs_torch_std_result_dim_1_1_0_max_0_03(self):
        # Arrange
        signal = torch.randn(10, 50)
        # Act
        result = scitex.dsp.norm.z(signal)
        # Act
        # Assert
        assert torch.abs(torch.std(result, dim=-1) - 1.0).max() < 0.03


    def test_preserves_shape_smoke_case_2(self):
        """Test that function preserves input shape."""
        # Arrange
        # Act
        # Assert
        shapes = [(10,), (5, 20), (3, 4, 50), (2, 3, 4, 100)]
        for shape in shapes:
            signal = np.random.randn(*shape)
            result = scitex.dsp.norm.z(signal)
            assert result.shape == signal.shape


class TestMinmax:
    """Test minmax normalization function."""

    def test_import_hasattr_scitex_dsp_norm_minmax(self):
        """Test function can be imported."""
        # Arrange
        # Act
        # Assert
        assert hasattr(scitex.dsp.norm, "minmax")

    def test_basic_1d_np_abs_result_max_pytest_approx_1_0(self):
        # Arrange
        signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert np.abs(result).max() == pytest.approx(1.0)

    def test_basic_1d_result_0_pytest_approx_1_0(self):
        # Arrange
        signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result[0] == pytest.approx(-1.0)

    def test_basic_1d_result_1_pytest_approx_1_0(self):
        # Arrange
        signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result[-1] == pytest.approx(1.0)


    def test_basic_2d_np_allclose_max_abs_per_row_1_0(self):
        """Test minmax normalization on 2D signal."""
        # Arrange
        signal = np.random.randn(5, 50) * 10
        result = scitex.dsp.norm.minmax(signal, dim=-1)

        # Check each row's max absolute value is 1.0
        # Act
        max_abs_per_row = np.abs(result).max(axis=-1)
        # Assert
        assert np.allclose(max_abs_per_row, 1.0)

    def test_basic_3d_result_shape_equals_signal_shape(self):
        # Arrange
        signal = np.random.randn(3, 4, 50) * 5
        # Act
        result = scitex.dsp.norm.minmax(signal, dim=-1)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_basic_3d_all_np_abs_row_max_pytest_approx_1_0_for_row_in_flat_result_result_shape_equals_signal_shape(self):
        # Arrange
        signal = np.random.randn(3, 4, 50) * 5
        # Act
        result = scitex.dsp.norm.minmax(signal, dim=-1)
        # Act
        # Assert
        assert result.shape == signal.shape

    def test_basic_3d_all_rows_max_abs_equals_one(self):
        """minmax on 3D signal: every flattened row's max abs equals 1.0."""
        # Arrange
        signal = np.random.randn(3, 4, 50) * 5
        # Act
        result = scitex.dsp.norm.minmax(signal, dim=-1)
        # Assert
        assert all(
            np.abs(row).max() == pytest.approx(1.0) for row in result.reshape(-1, 50)
        )



    def test_amplitude_scaling_np_abs_result_max_pytest_approx_2_0(self):
        # Arrange
        signal = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        # Test amp=2.0
        # Act
        result = scitex.dsp.norm.minmax(signal, amp=2.0)
        # Act
        # Assert
        assert np.abs(result).max() == pytest.approx(2.0)

    def test_amplitude_scaling_np_abs_result_max_pytest_approx_0_5_np_abs_result_max_pytest_approx_2_0(self):
        # Arrange
        signal = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        # Test amp=2.0
        # Act
        result = scitex.dsp.norm.minmax(signal, amp=2.0)
        # Act
        # Assert
        assert np.abs(result).max() == pytest.approx(2.0)

    def test_amplitude_scaling_amp_0_5_max_abs_equals_0_5(self):
        """minmax with amp=0.5 yields max abs == 0.5."""
        # Arrange
        signal = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        # Act
        result = scitex.dsp.norm.minmax(signal, amp=0.5)
        # Assert
        assert np.abs(result).max() == pytest.approx(0.5)



    def test_different_dimensions_smoke_case(self):
        """Test normalization along different dimensions."""
        # Arrange
        # Act
        # Assert
        signal = np.random.randn(4, 5, 6) * 3

        # Test along each dimension
        for dim in [0, 1, 2, -1]:
            result = scitex.dsp.norm.minmax(signal, dim=dim)
            # Get max absolute values along the specified dimension
            max_vals = np.abs(result).max(axis=dim)
            assert np.allclose(max_vals, 1.0)

    def test_positive_only_signal_result_max_pytest_approx_1_0(self):
        # Arrange
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result.max() == pytest.approx(1.0)

    def test_positive_only_signal_result_min_pytest_approx_0_2(self):
        # Arrange
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result.min() == pytest.approx(0.2)  # 1/5


    def test_negative_only_signal_result_min_pytest_approx_1_0(self):
        # Arrange
        signal = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result.min() == pytest.approx(-1.0)

    def test_negative_only_signal_result_max_pytest_approx_0_2(self):
        # Arrange
        signal = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result.max() == pytest.approx(-0.2)  # -1/5


    def test_symmetric_signal_result_min_pytest_approx_1_0(self):
        # Arrange
        signal = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result.min() == pytest.approx(-1.0)

    def test_symmetric_signal_result_max_pytest_approx_1_0(self):
        # Arrange
        signal = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result.max() == pytest.approx(1.0)

    def test_symmetric_signal_result_2_pytest_approx_0_0(self):
        # Arrange
        signal = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result[2] == pytest.approx(0.0)  # Middle value stays 0


    def test_torch_input_result_is_torch_tensor(self):
        # Arrange
        signal = torch.randn(5, 50) * 10
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_torch_allclose_max_abs_per_row_torch_ones_like_max_abs_per_r_result_is_torch_tensor(self):
        # Arrange
        signal = torch.randn(5, 50) * 10
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_torch_input_max_abs_per_row_equals_ones(self):
        """minmax on torch input yields max abs per row equal to 1.0."""
        # Arrange
        signal = torch.randn(5, 50) * 10
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Assert
        max_abs_per_row = torch.abs(result).max(dim=-1)[0]
        assert torch.allclose(max_abs_per_row, torch.ones_like(max_abs_per_row))



    def test_zero_signal_np_all_np_isnan_result_or_np_all_np_isinf_result_o(self):
        """Test with zero signal."""
        # Arrange
        signal = np.zeros((5, 10))
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Division by zero results in NaN values
        # Assert
        assert (
            np.all(np.isnan(result))
            or np.all(np.isinf(result))
            or np.allclose(result, 0)
        )

    def test_preserves_shape_smoke_case_2(self):
        """Test that function preserves input shape."""
        # Arrange
        # Act
        # Assert
        shapes = [(10,), (5, 20), (3, 4, 50), (2, 3, 4, 100)]
        for shape in shapes:
            signal = np.random.randn(*shape) * 5
            result = scitex.dsp.norm.minmax(signal)
            assert result.shape == signal.shape

    def test_edge_cases_result_0_pytest_approx_1_0(self):
        # Arrange
        signal = np.array([5.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result[0] == pytest.approx(1.0)

    def test_edge_cases_result_0_pytest_approx_1_0_result_0_pytest_approx_1_0(self):
        # Arrange
        signal = np.array([5.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Act
        # Assert
        assert result[0] == pytest.approx(1.0)

    def test_edge_cases_symmetric_pair_result_0_equals_neg_one(self):
        """minmax on [-2, 2]: first element normalizes to -1.0."""
        # Arrange
        signal = np.array([-2.0, 2.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Assert
        assert result[0] == pytest.approx(-1.0)

    def test_edge_cases_symmetric_pair_result_1_equals_one(self):
        """minmax on [-2, 2]: second element normalizes to +1.0."""
        # Arrange
        signal = np.array([-2.0, 2.0])
        # Act
        result = scitex.dsp.norm.minmax(signal)
        # Assert
        assert result[1] == pytest.approx(1.0)




if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/norm.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-04-05 12:15:42 (ywatanabe)"
#
# try:
#     import torch as _torch
#
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#     _torch = None
#
# from scitex.decorators import signal_fn as _signal_fn
#
#
# def _check_torch():
#     if not TORCH_AVAILABLE:
#         raise ImportError(
#             "PyTorch is not installed. Please install with: pip install torch"
#         )
#
#
# @_signal_fn
# def z(x, dim=-1):
#     _check_torch()
#     return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)
#
#
# @_signal_fn
# def minmax(x, amp=1.0, dim=-1, fn="mean"):
#     _check_torch()
#     MM = x.max(dim=dim, keepdims=True)[0].abs()
#     mm = x.min(dim=dim, keepdims=True)[0].abs()
#     return amp * x / _torch.maximum(MM, mm)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/norm.py
# --------------------------------------------------------------------------------
