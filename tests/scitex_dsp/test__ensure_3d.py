#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-01 20:45:00 (ywatanabe)"
# File: ./tests/scitex/dsp/test__ensure_3d.py

"""
Test module for scitex.dsp.ensure_3d function.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np


class TestEnsure3D:
    """Test class for ensure_3d function."""

    def test_import_callable_ensure_3d(self):
        """Test that ensure_3d can be imported."""
        # Arrange
        # Act
        from scitex.dsp import ensure_3d

        # Assert
        assert callable(ensure_3d)

    def test_1d_to_3d_numpy_result_shape_equals_n_1_1_5(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 1D input (seq_len,)
        x_1d = np.array([1, 2, 3, 4, 5])
        # Act
        result = ensure_3d(x_1d)
        # Act
        # Assert
        assert result.shape == (1, 1, 5)

    def test_1d_to_3d_numpy_np_array_equal_result_0_0_x_1d(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 1D input (seq_len,)
        x_1d = np.array([1, 2, 3, 4, 5])
        # Act
        result = ensure_3d(x_1d)
        # Act
        # Assert
        assert np.array_equal(result[0, 0], x_1d)


    def test_2d_to_3d_numpy_result_shape_equals_n_2_1_3(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 2D input (batch_size, seq_len)
        x_2d = np.array([[1, 2, 3], [4, 5, 6]])
        # Act
        result = ensure_3d(x_2d)
        # Act
        # Assert
        assert result.shape == (2, 1, 3)

    def test_2d_to_3d_numpy_np_array_equal_result_0_x_2d(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 2D input (batch_size, seq_len)
        x_2d = np.array([[1, 2, 3], [4, 5, 6]])
        # Act
        result = ensure_3d(x_2d)
        # Act
        # Assert
        assert np.array_equal(result[:, 0, :], x_2d)


    def test_3d_unchanged_numpy_result_shape_equals_x_3d_shape(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 3D input (batch_size, n_channels, seq_len)
        x_3d = np.random.rand(4, 3, 100)
        # Act
        result = ensure_3d(x_3d)
        # Act
        # Assert
        assert result.shape == x_3d.shape

    def test_3d_unchanged_numpy_np_array_equal_result_x_3d(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 3D input (batch_size, n_channels, seq_len)
        x_3d = np.random.rand(4, 3, 100)
        # Act
        result = ensure_3d(x_3d)
        # Act
        # Assert
        assert np.array_equal(result, x_3d)


    def test_1d_to_3d_torch_result_is_torch_tensor(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 1D input (seq_len,)
        x_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # Act
        result = ensure_3d(x_1d)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_1d_to_3d_torch_result_shape_equals_n_1_1_5(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 1D input (seq_len,)
        x_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # Act
        result = ensure_3d(x_1d)
        # Act
        # Assert
        assert result.shape == (1, 1, 5)

    def test_1d_to_3d_torch_torch_equal_result_0_0_x_1d(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 1D input (seq_len,)
        x_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # Act
        result = ensure_3d(x_1d)
        # Act
        # Assert
        assert torch.equal(result[0, 0], x_1d)


    def test_2d_to_3d_torch_result_is_torch_tensor(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 2D input (batch_size, seq_len)
        x_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Act
        result = ensure_3d(x_2d)
        # Act
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_2d_to_3d_torch_result_shape_equals_n_2_1_3(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 2D input (batch_size, seq_len)
        x_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Act
        result = ensure_3d(x_2d)
        # Act
        # Assert
        assert result.shape == (2, 1, 3)

    def test_2d_to_3d_torch_torch_equal_result_0_x_2d(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 2D input (batch_size, seq_len)
        x_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Act
        result = ensure_3d(x_2d)
        # Act
        # Assert
        assert torch.equal(result[:, 0, :], x_2d)


    def test_3d_unchanged_torch_result_shape_equals_x_3d_shape(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 3D input (batch_size, n_channels, seq_len)
        x_3d = torch.randn(4, 3, 100)
        # Act
        result = ensure_3d(x_3d)
        # Act
        # Assert
        assert result.shape == x_3d.shape

    def test_3d_unchanged_torch_torch_equal_result_x_3d(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # 3D input (batch_size, n_channels, seq_len)
        x_3d = torch.randn(4, 3, 100)
        # Act
        result = ensure_3d(x_3d)
        # Act
        # Assert
        assert torch.equal(result, x_3d)


    def test_empty_arrays_result_shape_equals_n_1_1_0(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # Empty 1D
        x_empty_1d = np.array([])
        # Act
        result = ensure_3d(x_empty_1d)
        # Act
        # Assert
        assert result.shape == (1, 1, 0)

    def test_empty_arrays_result_shape_1_1_result_shape_equals_n_1_1_0(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # Empty 1D
        x_empty_1d = np.array([])
        # Act
        result = ensure_3d(x_empty_1d)
        # Act
        # Assert
        assert result.shape == (1, 1, 0)

    def test_empty_arrays_empty_2d_adds_channel_dimension(self):
        # Arrange
        from scitex.dsp import ensure_3d
        x_empty_2d = np.array([[], []])
        # Act
        result = ensure_3d(x_empty_2d)
        # Assert
        assert result.shape[1] == 1  # Added channel dimension



    def test_single_element_result_shape_equals_n_1_1_1(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # Single element 1D
        x_single = np.array([42])
        # Act
        result = ensure_3d(x_single)
        # Act
        # Assert
        assert result.shape == (1, 1, 1)

    def test_single_element_result_0_0_0_42(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # Single element 1D
        x_single = np.array([42])
        # Act
        result = ensure_3d(x_single)
        # Act
        # Assert
        assert result[0, 0, 0] == 42


    def test_large_arrays_result_shape_equals_n_1_1_10000(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # Large 1D array
        x_large_1d = np.random.rand(10000)
        # Act
        result = ensure_3d(x_large_1d)
        # Act
        # Assert
        assert result.shape == (1, 1, 10000)

    def test_large_arrays_result_shape_equals_n_100_1_1000_result_shape_equals_n_1_1_10000(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # Large 1D array
        x_large_1d = np.random.rand(10000)
        # Act
        result = ensure_3d(x_large_1d)
        # Act
        # Assert
        assert result.shape == (1, 1, 10000)

    def test_large_arrays_2d_inserts_channel_axis(self):
        # Arrange
        from scitex.dsp import ensure_3d
        x_large_2d = np.random.rand(100, 1000)
        # Act
        result = ensure_3d(x_large_2d)
        # Assert
        assert result.shape == (100, 1, 1000)



    def test_dtype_preservation_smoke_case(self):
        """Test that floating dtypes are preserved.

        ``signal_fn`` (the underlying decorator) only restores *floating*
        dtypes on the way out — integer inputs get promoted to the
        decorator's working float dtype because signal-processing kernels
        typically require float.
        """
        # Arrange
        # Act
        # Assert
        from scitex.dsp import ensure_3d

        for dtype in [np.float32, np.float64]:
            x = np.array([1, 2, 3], dtype=dtype)
            result = ensure_3d(x)
            assert result.dtype == dtype

    def test_torch_device_preservation_cpu(self):
        """ensure_3d preserves CPU device on CPU tensors."""
        # Arrange
        from scitex.dsp import ensure_3d
        x_cpu = torch.tensor([1.0, 2.0, 3.0])
        # Act
        result = ensure_3d(x_cpu)
        # Assert
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_device_preservation_cuda_device_equals_input(self):
        """ensure_3d preserves CUDA device identity on CUDA tensors."""
        # Arrange
        from scitex.dsp import ensure_3d
        x_gpu = torch.tensor([1.0, 2.0, 3.0]).cuda()
        # Act
        result = ensure_3d(x_gpu)
        # Assert
        assert result.device == x_gpu.device

    def test_gradient_preservation_result_requires_grad(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # Create tensor with gradient
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        # Act
        result = ensure_3d(x)
        # Act
        # Assert
        assert result.requires_grad

    def test_gradient_preservation_x_grad_is_not_none_result_requires_grad(self):
        # Arrange
        from scitex.dsp import ensure_3d
        # Create tensor with gradient
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        # Act
        result = ensure_3d(x)
        # Act
        # Assert
        assert result.requires_grad

    def test_gradient_preservation_backward_populates_input_grad(self):
        # Arrange
        from scitex.dsp import ensure_3d
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = ensure_3d(x)
        # Act
        loss = result.sum()
        loss.backward()
        # Assert
        assert x.grad is not None



    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((5,), (1, 1, 5)),  # 1D
            ((3, 10), (3, 1, 10)),  # 2D
            ((2, 4, 8), (2, 4, 8)),  # 3D
        ],
    )
    def test_various_shapes_result_shape_equals_expected(self, shape, expected):
        """Test various input shapes."""
        # Arrange
        from scitex.dsp import ensure_3d

        x = np.ones(shape)
        # Act
        result = ensure_3d(x)
        # Assert
        assert result.shape == expected

    def test_list_input_result_is_list(self):
        # Arrange
        from scitex.dsp import ensure_3d
        x_list = [1, 2, 3, 4, 5]
        # Act
        result = ensure_3d(x_list)
        # Act
        # Assert
        assert isinstance(result, list)

    def test_list_input_len_result_is_1(self):
        # Arrange
        from scitex.dsp import ensure_3d
        x_list = [1, 2, 3, 4, 5]
        # Act
        result = ensure_3d(x_list)
        # Act
        # Assert
        assert len(result) == 1

    def test_list_input_len_result_0_is_1(self):
        # Arrange
        from scitex.dsp import ensure_3d
        x_list = [1, 2, 3, 4, 5]
        # Act
        result = ensure_3d(x_list)
        # Act
        # Assert
        assert len(result[0]) == 1

    def test_list_input_len_result_0_0_is_5(self):
        # Arrange
        from scitex.dsp import ensure_3d
        x_list = [1, 2, 3, 4, 5]
        # Act
        result = ensure_3d(x_list)
        # Act
        # Assert
        assert len(result[0][0]) == 5

    def test_list_input_np_array_equal_result_0_0_x_list(self):
        # Arrange
        from scitex.dsp import ensure_3d
        x_list = [1, 2, 3, 4, 5]
        # Act
        result = ensure_3d(x_list)
        # Act
        # Assert
        assert np.array_equal(result[0][0], x_list)



if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_ensure_3d.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 01:03:47 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/_ensure_3d.py
#
# from scitex.decorators import signal_fn
#
#
# @signal_fn
# def ensure_3d(x):
#     if x.ndim == 1:  # assumes (seq_len,)
#         x = x.unsqueeze(0).unsqueeze(0)
#     elif x.ndim == 2:  # assumes (batch_siize, seq_len)
#         x = x.unsqueeze(1)
#     return x
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_ensure_3d.py
# --------------------------------------------------------------------------------
