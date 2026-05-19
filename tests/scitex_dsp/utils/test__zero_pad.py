#!/usr/bin/env python3
# Time-stamp: "2025-06-02 15:30:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/utils/test__zero_pad.py

"""Tests for zero padding utilities."""

import os

import pytest

torch = pytest.importorskip("torch")

import numpy as np


class TestZeroPadAvailableFlags:
    """Test _AVAILABLE flags for optional dependencies."""

    def test_torch_available_flag_exists(self):
        """Test that TORCH_AVAILABLE flag is exported."""
        # Arrange
        # Act
        from scitex.dsp.utils._zero_pad import TORCH_AVAILABLE

        # Assert
        assert isinstance(TORCH_AVAILABLE, bool)

    def test_check_torch_function_exists(self):
        """Test that _check_torch function is exported."""
        # Arrange
        # Act
        from scitex.dsp.utils._zero_pad import _check_torch

        # Assert
        assert callable(_check_torch)

    def test_torch_available_is_true_when_torch_installed(self):
        """Test that TORCH_AVAILABLE is True when torch is installed."""
        # Arrange
        # Act
        from scitex.dsp.utils._zero_pad import TORCH_AVAILABLE

        # Assert
        assert TORCH_AVAILABLE is True

    def test_check_torch_does_not_raise_when_available(self):
        """Test that _check_torch doesn't raise when torch is available."""
        # Arrange
        # Act
        # Assert
        from scitex.dsp.utils._zero_pad import _check_torch

        _check_torch()


def test_zero_pad_1d_basic_len_result_is_7():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2, 3])
    # Act
    result = _zero_pad_1d(x, 7)
    # Act
    # Assert
    assert len(result) == 7


def test_zero_pad_1d_basic_torch_equal_result_2_5_x():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2, 3])
    # Act
    result = _zero_pad_1d(x, 7)
    # Act
    # Assert
    assert torch.equal(result[2:5], x)


def test_zero_pad_1d_basic_torch_equal_result_2_torch_zeros_2():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2, 3])
    # Act
    result = _zero_pad_1d(x, 7)
    # Act
    # Assert
    assert torch.equal(result[:2], torch.zeros(2))


def test_zero_pad_1d_basic_torch_equal_result_5_torch_zeros_2():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2, 3])
    # Act
    result = _zero_pad_1d(x, 7)
    # Act
    # Assert
    assert torch.equal(result[5:], torch.zeros(2))




def test_zero_pad_1d_even_padding_len_result_is_6():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2])
    # Act
    result = _zero_pad_1d(x, 6)
    # Act
    # Assert
    assert len(result) == 6


def test_zero_pad_1d_even_padding_torch_equal_result_2_4_x():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2])
    # Act
    result = _zero_pad_1d(x, 6)
    # Act
    # Assert
    assert torch.equal(result[2:4], x)


def test_zero_pad_1d_even_padding_torch_equal_result_2_torch_zeros_2():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2])
    # Act
    result = _zero_pad_1d(x, 6)
    # Act
    # Assert
    assert torch.equal(result[:2], torch.zeros(2))


def test_zero_pad_1d_even_padding_torch_equal_result_4_torch_zeros_2():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2])
    # Act
    result = _zero_pad_1d(x, 6)
    # Act
    # Assert
    assert torch.equal(result[4:], torch.zeros(2))




def test_zero_pad_1d_odd_padding_len_result_is_5():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2])
    # Act
    result = _zero_pad_1d(x, 5)
    # Act
    # Assert
    assert len(result) == 5


def test_zero_pad_1d_odd_padding_torch_equal_result_1_3_x():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2])
    # Act
    result = _zero_pad_1d(x, 5)
    # Act
    # Assert
    assert torch.equal(result[1:3], x)


def test_zero_pad_1d_odd_padding_torch_equal_result_1_torch_zeros_1():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2])
    # Act
    result = _zero_pad_1d(x, 5)
    # Act
    # Assert
    assert torch.equal(result[:1], torch.zeros(1))


def test_zero_pad_1d_odd_padding_torch_equal_result_3_torch_zeros_2():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = torch.tensor([1, 2])
    # Act
    result = _zero_pad_1d(x, 5)
    # Act
    # Assert
    assert torch.equal(result[3:], torch.zeros(2))




def test_zero_pad_1d_no_padding_needed():
    """Test 1D zero padding when no padding is needed."""
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d

    x = torch.tensor([1, 2, 3, 4, 5])
    # Act
    result = _zero_pad_1d(x, 5)

    # Should be unchanged
    # Assert
    assert torch.equal(result, x)


def test_zero_pad_1d_numpy_input_result_is_torch_tensor():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = np.array([1, 2, 3])
    # Act
    result = _zero_pad_1d(x, 7)
    # Act
    # Assert
    assert isinstance(result, torch.Tensor)


def test_zero_pad_1d_numpy_input_len_result_is_7():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = np.array([1, 2, 3])
    # Act
    result = _zero_pad_1d(x, 7)
    # Act
    # Assert
    assert len(result) == 7


def test_zero_pad_1d_numpy_input_torch_equal_result_2_5_torch_tensor_1_2_3():
    # Arrange
    from scitex.dsp.utils import _zero_pad_1d
    x = np.array([1, 2, 3])
    # Act
    result = _zero_pad_1d(x, 7)
    # Act
    # Assert
    assert torch.equal(result[2:5], torch.tensor([1, 2, 3]))




def test_zero_pad_basic_result_shape_equals_n_3_4():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2, 3])
    x2 = torch.tensor([4, 5])
    x3 = torch.tensor([6, 7, 8, 9])
    # Act
    result = zero_pad([x1, x2, x3])
    # Act
    # Assert
    assert result.shape == (3, 4)


def test_zero_pad_basic_torch_equal_result_0_torch_tensor_1_2_3_0():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2, 3])
    x2 = torch.tensor([4, 5])
    x3 = torch.tensor([6, 7, 8, 9])
    # Act
    result = zero_pad([x1, x2, x3])
    # Act
    # Assert
    assert torch.equal(result[0], torch.tensor([1, 2, 3, 0]))


def test_zero_pad_basic_torch_equal_result_1_torch_tensor_0_4_5_0():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2, 3])
    x2 = torch.tensor([4, 5])
    x3 = torch.tensor([6, 7, 8, 9])
    # Act
    result = zero_pad([x1, x2, x3])
    # Act
    # Assert
    assert torch.equal(result[1], torch.tensor([0, 4, 5, 0]))


def test_zero_pad_basic_torch_equal_result_2_torch_tensor_6_7_8_9():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2, 3])
    x2 = torch.tensor([4, 5])
    x3 = torch.tensor([6, 7, 8, 9])
    # Act
    result = zero_pad([x1, x2, x3])
    # Act
    # Assert
    assert torch.equal(result[2], torch.tensor([6, 7, 8, 9]))




def test_zero_pad_mixed_inputs_result_shape_equals_n_3_5():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2, 3])  # torch tensor
    x2 = np.array([4, 5])  # numpy array
    x3 = [6, 7, 8, 9, 10]  # list
    # Act
    result = zero_pad([x1, x2, x3])
    # Act
    # Assert
    assert result.shape == (3, 5)


def test_zero_pad_mixed_inputs_result_is_torch_tensor():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2, 3])  # torch tensor
    x2 = np.array([4, 5])  # numpy array
    x3 = [6, 7, 8, 9, 10]  # list
    # Act
    result = zero_pad([x1, x2, x3])
    # Act
    # Assert
    assert isinstance(result, torch.Tensor)


def test_zero_pad_mixed_inputs_torch_equal_result_0_torch_tensor_0_1_2_3_0():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2, 3])  # torch tensor
    x2 = np.array([4, 5])  # numpy array
    x3 = [6, 7, 8, 9, 10]  # list
    # Act
    result = zero_pad([x1, x2, x3])
    # Act
    # Assert
    assert torch.equal(result[0], torch.tensor([0, 1, 2, 3, 0]))


def test_zero_pad_mixed_inputs_torch_equal_result_1_torch_tensor_0_4_5_0_0():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2, 3])  # torch tensor
    x2 = np.array([4, 5])  # numpy array
    x3 = [6, 7, 8, 9, 10]  # list
    # Act
    result = zero_pad([x1, x2, x3])
    # Act
    # Assert
    assert torch.equal(result[1], torch.tensor([0, 4, 5, 0, 0]))


def test_zero_pad_mixed_inputs_torch_equal_result_2_torch_tensor_6_7_8_9_10():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2, 3])  # torch tensor
    x2 = np.array([4, 5])  # numpy array
    x3 = [6, 7, 8, 9, 10]  # list
    # Act
    result = zero_pad([x1, x2, x3])
    # Act
    # Assert
    assert torch.equal(result[2], torch.tensor([6, 7, 8, 9, 10]))




def test_zero_pad_single_tensor_result_shape_equals_n_1_4():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x = torch.tensor([1, 2, 3, 4])
    # Act
    result = zero_pad([x])
    # Act
    # Assert
    assert result.shape == (1, 4)


def test_zero_pad_single_tensor_torch_equal_result_0_x():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x = torch.tensor([1, 2, 3, 4])
    # Act
    result = zero_pad([x])
    # Act
    # Assert
    assert torch.equal(result[0], x)




def test_zero_pad_empty_list():
    """Test zero padding with empty list."""
    # Arrange
    # Act
    from scitex.dsp.utils import zero_pad

    # Assert
    with pytest.raises(ValueError):
        zero_pad([])


def test_zero_pad_different_dimensions_result_dim0_shape_equals_n_2_3():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2])
    x2 = torch.tensor([3, 4, 5])
    # Test dim=0 (default): two 1-D tensors stacked along new axis 0
    # → shape (n_inputs, max_len) = (2, 3)
    # Act
    result_dim0 = zero_pad([x1, x2], dim=0)
    # Act
    # Assert
    assert result_dim0.shape == (2, 3)


def test_zero_pad_different_dimensions_result_dim1_shape_equals_n_3_2_result_dim0_shape_equals_n_2_3():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2])
    x2 = torch.tensor([3, 4, 5])
    # Test dim=0 (default): two 1-D tensors stacked along new axis 0
    # → shape (n_inputs, max_len) = (2, 3)
    # Act
    result_dim0 = zero_pad([x1, x2], dim=0)
    # Act
    # Assert
    assert result_dim0.shape == (2, 3)


def test_zero_pad_different_dimensions_result_dim1_shape_equals_n_3_2_result_dim1_shape_equals_n_3_2():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1, 2])
    x2 = torch.tensor([3, 4, 5])
    # Test dim=0 (default): two 1-D tensors stacked along new axis 0
    # → shape (n_inputs, max_len) = (2, 3)
    # Act
    result_dim0 = zero_pad([x1, x2], dim=0)
    # Assert
    assert result_dim0.shape == (2, 3)
    # Test dim=1: stacked along axis 1 → shape (max_len, n_inputs) = (3, 2).
    # torch.stack(.., dim=1) inserts the new axis AT 1, not at 0.
    result_dim1 = zero_pad([x1, x2], dim=1)
    # Act
    # Assert
    assert result_dim1.shape == (3, 2)






def test_zero_pad_preserve_dtype():
    """Test that zero padding preserves data types."""
    # Arrange
    # Act
    # Assert
    from scitex.dsp.utils import zero_pad

    # Test different dtypes
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

    for dtype in dtypes:
        x1 = torch.tensor([1, 2], dtype=dtype)
        x2 = torch.tensor([3, 4, 5], dtype=dtype)

        result = zero_pad([x1, x2])
        assert result.dtype == dtype


def test_zero_pad_preserve_device():
    """Test that zero padding preserves device."""
    # Arrange
    from scitex.dsp.utils import zero_pad

    x1 = torch.tensor([1, 2])
    x2 = torch.tensor([3, 4, 5])

    # Act
    result = zero_pad([x1, x2])
    # Assert
    assert result.device == x1.device

    # Test CUDA if available
    if torch.cuda.is_available():
        x1_cuda = x1.cuda()
        x2_cuda = x2.cuda()
        result_cuda = zero_pad([x1_cuda, x2_cuda])
        assert result_cuda.device == x1_cuda.device


def test_zero_pad_large_size_difference_result_shape_equals_n_2_100():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1])  # length 1
    x2 = torch.tensor(list(range(100)))  # length 100
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert result.shape == (2, 100)


def test_zero_pad_large_size_difference_result_0_49_1():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1])  # length 1
    x2 = torch.tensor(list(range(100)))  # length 100
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert result[0, 49] == 1  # Should be in the middle


def test_zero_pad_large_size_difference_torch_sum_result_0_0_99():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1])  # length 1
    x2 = torch.tensor(list(range(100)))  # length 100
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert torch.sum(result[0] == 0) == 99  # 99 zeros


def test_zero_pad_large_size_difference_torch_equal_result_1_x2():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1])  # length 1
    x2 = torch.tensor(list(range(100)))  # length 100
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert torch.equal(result[1], x2)




def test_zero_pad_real_signal_example():
    """Test zero padding with realistic signal processing example."""
    # Arrange
    from scitex.dsp.utils import zero_pad

    # Simulate different length EEG trials
    fs = 250  # Hz
    trial1 = torch.sin(2 * np.pi * 10 * torch.linspace(0, 1, fs))  # 1 second
    trial2 = torch.sin(
        2 * np.pi * 10 * torch.linspace(0, 1.5, int(fs * 1.5))
    )  # 1.5 seconds
    trial3 = torch.sin(2 * np.pi * 10 * torch.linspace(0, 2, fs * 2))  # 2 seconds

    trials = [trial1, trial2, trial3]
    # Act
    result = zero_pad(trials)

    # Should be padded to longest trial (2 seconds = 500 samples)
    # Assert
    assert result.shape == (3, 500)

    # Check that original signals are preserved (centered-padding offsets)
    # rather than scanning for nonzero — sine waves cross zero at interior
    # points and the heuristic mis-detects boundaries.
    max_len = result.shape[1]
    for i, trial in enumerate(trials):
        L = trial.shape[0]
        needed = max_len - L
        left = needed // 2
        extracted = result[i, left : left + L]
        assert extracted.shape == trial.shape
        assert torch.allclose(extracted, trial, atol=1e-6)


def test_zero_pad_gradient_flow_x1_grad_is_not_none():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1.0, 2.0], requires_grad=True)
    x2 = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)
    result = zero_pad([x1, x2])
    loss = result.sum()
    # Act
    loss.backward()
    # Act
    # Assert
    assert x1.grad is not None


def test_zero_pad_gradient_flow_x2_grad_is_not_none():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1.0, 2.0], requires_grad=True)
    x2 = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)
    result = zero_pad([x1, x2])
    loss = result.sum()
    # Act
    loss.backward()
    # Act
    # Assert
    assert x2.grad is not None


def test_zero_pad_gradient_flow_torch_equal_x1_grad_torch_ones_like_x1():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1.0, 2.0], requires_grad=True)
    x2 = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)
    result = zero_pad([x1, x2])
    loss = result.sum()
    # Act
    loss.backward()
    # Act
    # Assert
    assert torch.equal(x1.grad, torch.ones_like(x1))


def test_zero_pad_gradient_flow_torch_equal_x2_grad_torch_ones_like_x2():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.tensor([1.0, 2.0], requires_grad=True)
    x2 = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)
    result = zero_pad([x1, x2])
    loss = result.sum()
    # Act
    loss.backward()
    # Act
    # Assert
    assert torch.equal(x2.grad, torch.ones_like(x2))




def test_zero_pad_empty_tensors_result_shape_equals_n_2_3():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.empty(0)
    x2 = torch.tensor([1, 2, 3])
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert result.shape == (2, 3)


def test_zero_pad_empty_tensors_torch_equal_result_0_torch_zeros_3():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.empty(0)
    x2 = torch.tensor([1, 2, 3])
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert torch.equal(result[0], torch.zeros(3))


def test_zero_pad_empty_tensors_torch_equal_result_1_x2():
    # Arrange
    from scitex.dsp.utils import zero_pad
    x1 = torch.empty(0)
    x2 = torch.tensor([1, 2, 3])
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert torch.equal(result[1], x2)




def test_zero_pad_consistency():
    """Test that zero padding is consistent across calls."""
    # Arrange
    from scitex.dsp.utils import zero_pad

    x1 = torch.tensor([1, 2])
    x2 = torch.tensor([3, 4, 5])

    result1 = zero_pad([x1, x2])
    # Act
    result2 = zero_pad([x1, x2])

    # Assert
    assert torch.equal(result1, result2)


def test_zero_pad_different_numeric_types_result_int_dtype_in_torch_int64_torch_long():
    # Arrange
    from scitex.dsp.utils import zero_pad
    # Test with integers
    x1 = [1, 2]
    x2 = [3, 4, 5]
    # Act
    result_int = zero_pad([x1, x2])
    # Act
    # Assert
    assert result_int.dtype in [torch.int64, torch.long]  # Default int type


def test_zero_pad_different_numeric_types_result_float_dtype_in_torch_float32_torch_float_result_int_dtype_in_torch_int64_torch_long():
    # Arrange
    from scitex.dsp.utils import zero_pad
    # Test with integers
    x1 = [1, 2]
    x2 = [3, 4, 5]
    # Act
    result_int = zero_pad([x1, x2])
    # Act
    # Assert
    assert result_int.dtype in [torch.int64, torch.long]  # Default int type


def test_zero_pad_different_numeric_types_result_float_dtype_in_torch_float32_torch_float_result_float_dtype_in_torch_float32_torch_float():
    # Arrange
    from scitex.dsp.utils import zero_pad
    # Test with integers
    x1 = [1, 2]
    x2 = [3, 4, 5]
    # Act
    result_int = zero_pad([x1, x2])
    # Assert
    assert result_int.dtype in [torch.int64, torch.long]  # Default int type
    # Test with floats
    x1 = [1.0, 2.0]
    x2 = [3.0, 4.0, 5.0]
    result_float = zero_pad([x1, x2])
    # Act
    # Assert
    assert result_float.dtype in [torch.float32, torch.float64]  # Default float type






def test_zero_pad_edge_cases_result_shape_equals_n_2_1():
    # Arrange
    from scitex.dsp.utils import zero_pad
    # Test with single element tensors
    x1 = torch.tensor([42])
    x2 = torch.tensor([99])
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert result.shape == (2, 1)


def test_zero_pad_edge_cases_torch_equal_result_torch_tensor_42_99():
    # Arrange
    from scitex.dsp.utils import zero_pad
    # Test with single element tensors
    x1 = torch.tensor([42])
    x2 = torch.tensor([99])
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert torch.equal(result, torch.tensor([[42], [99]]))




def test_zero_pad_memory_efficiency_result_shape_equals_n_2_1000():
    # Arrange
    from scitex.dsp.utils import zero_pad
    # Create large tensors
    x1 = torch.randn(1000)
    x2 = torch.randn(1000)  # Same length, no padding needed
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert result.shape == (2, 1000)


def test_zero_pad_memory_efficiency_torch_equal_result_0_x1():
    # Arrange
    from scitex.dsp.utils import zero_pad
    # Create large tensors
    x1 = torch.randn(1000)
    x2 = torch.randn(1000)  # Same length, no padding needed
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert torch.equal(result[0], x1)


def test_zero_pad_memory_efficiency_torch_equal_result_1_x2():
    # Arrange
    from scitex.dsp.utils import zero_pad
    # Create large tensors
    x1 = torch.randn(1000)
    x2 = torch.randn(1000)  # Same length, no padding needed
    # Act
    result = zero_pad([x1, x2])
    # Act
    # Assert
    assert torch.equal(result[1], x2)




if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/_zero_pad.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-11-26 10:30:34 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/utils/_zero_pad.py
#
# THIS_FILE = "/home/ywatanabe/proj/scitex_repo/src/scitex/dsp/utils/_zero_pad.py"
#
# import numpy as np
#
# try:
#     import torch
#     import torch.nn.functional as F
#
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#     torch = None
#     F = None
#
#
# def _check_torch():
#     if not TORCH_AVAILABLE:
#         raise ImportError(
#             "PyTorch is not installed. Please install with: pip install torch"
#         )
#
#
# def _zero_pad_1d(x, target_length):
#     """Zero pad a 1D tensor to target length."""
#     _check_torch()
#     if not isinstance(x, torch.Tensor):
#         x = torch.tensor(x)
#     padding_needed = target_length - len(x)
#     padding_left = padding_needed // 2
#     padding_right = padding_needed - padding_left
#     return F.pad(x, (padding_left, padding_right), "constant", 0)
#
#
# def zero_pad(xs, dim=0):
#     """Zero pad a list of arrays to the same length.
#
#     Args:
#         xs: List of tensors or arrays
#         dim: Dimension to stack along
#
#     Returns:
#         Stacked tensor with zero padding
#     """
#     # Convert to tensors if needed
#     tensors = []
#     for x in xs:
#         if isinstance(x, np.ndarray):
#             tensors.append(torch.tensor(x))
#         elif isinstance(x, torch.Tensor):
#             tensors.append(x)
#         else:
#             tensors.append(torch.tensor(x))
#
#     max_len = max([len(x) for x in tensors])
#     return torch.stack([_zero_pad_1d(x, max_len) for x in tensors], dim=dim)
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/_zero_pad.py
# --------------------------------------------------------------------------------
