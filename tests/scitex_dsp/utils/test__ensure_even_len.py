#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:40:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/utils/test__ensure_even_len.py

"""Tests for ensure_even_len function."""

import os

import pytest

torch = pytest.importorskip("torch")
import numpy as np


def test_ensure_even_len_already_even_torch_equal_result_x():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(100)  # Even length
    # Act
    result = ensure_even_len(x)
    # Assert
    assert torch.equal(result, x)


def test_ensure_even_len_already_even_result_shape_equals_x_shape():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(100)  # Even length
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape == x.shape


def test_ensure_even_len_already_even_result_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(100)  # Even length
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_odd_to_even_result_shape_equals_n_100():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape == (100,)


def test_ensure_even_len_odd_to_even_result_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_odd_to_even_torch_equal_result_x_truncated():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Assert
    assert torch.equal(result, x[:-1])


def test_ensure_even_len_2d_tensor_result_shape_equals_n_10_100():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(10, 101)  # (batch, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape == (10, 100)


def test_ensure_even_len_2d_tensor_result_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(10, 101)  # (batch, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_2d_tensor_torch_equal_result_x_truncated():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(10, 101)  # (batch, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert torch.equal(result, x[:, :-1])


def test_ensure_even_len_3d_tensor_result_shape_equals_n_5_8_98():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(5, 8, 99)  # (batch, channels, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape == (5, 8, 98)


def test_ensure_even_len_3d_tensor_result_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(5, 8, 99)  # (batch, channels, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_3d_tensor_torch_equal_result_x_truncated():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(5, 8, 99)  # (batch, channels, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert torch.equal(result, x[:, :, :-1])


def test_ensure_even_len_4d_tensor_result_shape_equals_n_2_3_4_50():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(2, 3, 4, 51)  # Odd last dimension
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape == (2, 3, 4, 50)


def test_ensure_even_len_4d_tensor_result_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(2, 3, 4, 51)  # Odd last dimension
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_4d_tensor_torch_equal_result_x_truncated():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(2, 3, 4, 51)  # Odd last dimension
    # Act
    result = ensure_even_len(x)
    # Assert
    assert torch.equal(result, x[:, :, :, :-1])


def test_ensure_even_len_numpy_array_result_shape_equals_n_100():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = np.random.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape == (100,)


def test_ensure_even_len_numpy_array_result_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = np.random.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_numpy_array_np_array_equal_result_x_truncated():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = np.random.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Assert
    assert np.array_equal(result, x[:-1])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_ensure_even_len_preserves_dtype(dtype):
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101).to(dtype)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.dtype == dtype


def test_ensure_even_len_preserves_device_cpu():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x_cpu = torch.randn(101)
    # Act
    result_cpu = ensure_even_len(x_cpu)
    # Assert
    assert result_cpu.device == x_cpu.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ensure_even_len_preserves_device_cuda():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x_cuda = torch.randn(101).cuda()
    # Act
    result_cuda = ensure_even_len(x_cuda)
    # Assert
    assert result_cuda.device == x_cuda.device


def test_ensure_even_len_preserves_requires_grad_true():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101, requires_grad=True)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.requires_grad == x.requires_grad


def test_ensure_even_len_preserves_requires_grad_false():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101, requires_grad=False)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.requires_grad == x.requires_grad


def test_ensure_even_len_gradient_flow_x_grad_is_not_none():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101, requires_grad=True)
    result = ensure_even_len(x)
    loss = result.sum()
    # Act
    loss.backward()
    # Assert
    assert x.grad is not None


def test_ensure_even_len_gradient_flow_torch_equal_x_grad_100_torch_ones_100():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101, requires_grad=True)
    result = ensure_even_len(x)
    loss = result.sum()
    # Act
    loss.backward()
    # Assert
    assert torch.equal(x.grad[:100], torch.ones(100))


def test_ensure_even_len_gradient_flow_x_grad_last_zero():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101, requires_grad=True)
    result = ensure_even_len(x)
    loss = result.sum()
    # Act
    loss.backward()
    # Assert
    assert x.grad[100] == 0  # Last element should have zero gradient


def test_ensure_even_len_small_tensors_length_one_becomes_empty():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x1 = torch.tensor([5.0])
    # Act
    result1 = ensure_even_len(x1)
    # Assert
    assert result1.shape == (0,)


def test_ensure_even_len_small_tensors_length_two_unchanged():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x2 = torch.tensor([1.0, 2.0])
    # Act
    result2 = ensure_even_len(x2)
    # Assert
    assert torch.equal(result2, x2)


def test_ensure_even_len_small_tensors_length_three_truncates_to_two():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x3 = torch.tensor([1.0, 2.0, 3.0])
    # Act
    result3 = ensure_even_len(x3)
    # Assert
    assert torch.equal(result3, torch.tensor([1.0, 2.0]))


def test_ensure_even_len_empty_tensor_torch_equal_result_x():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.empty(0)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert torch.equal(result, x)


def test_ensure_even_len_empty_tensor_result_shape_equals_n_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.empty(0)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape == (0,)


def test_ensure_even_len_consistency_across_calls():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101)
    result1 = ensure_even_len(x)
    # Act
    result2 = ensure_even_len(x)
    # Assert
    assert torch.equal(result1, result2)


def test_ensure_even_len_real_signal_example_signal_is_odd_length():
    # Arrange
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    t = torch.linspace(0, duration, n_samples)
    # Act
    signal = torch.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    # Assert
    assert signal.shape[-1] % 2 == 1  # Should be odd


def test_ensure_even_len_real_signal_example_result_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    t = torch.linspace(0, duration, n_samples)
    signal = torch.sin(2 * np.pi * 10 * t)
    # Act
    result = ensure_even_len(signal)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_real_signal_example_result_shape_is_500():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    t = torch.linspace(0, duration, n_samples)
    signal = torch.sin(2 * np.pi * 10 * t)
    # Act
    result = ensure_even_len(signal)
    # Assert
    assert result.shape[-1] == 500  # One less than original


def test_ensure_even_len_real_signal_example_torch_allclose_result_signal_truncated():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    t = torch.linspace(0, duration, n_samples)
    signal = torch.sin(2 * np.pi * 10 * t)
    # Act
    result = ensure_even_len(signal)
    # Assert
    assert torch.allclose(result, signal[:-1])


def test_ensure_even_len_multichannel_signal_result_shape_equals_n_channels_1000():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    n_channels = 64
    n_samples = 1001  # Odd number of samples
    signal = torch.randn(n_channels, n_samples)
    # Act
    result = ensure_even_len(signal)
    # Assert
    assert result.shape == (n_channels, 1000)


def test_ensure_even_len_multichannel_signal_result_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    n_channels = 64
    n_samples = 1001  # Odd number of samples
    signal = torch.randn(n_channels, n_samples)
    # Act
    result = ensure_even_len(signal)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_multichannel_signal_torch_equal_result_signal_truncated():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    n_channels = 64
    n_samples = 1001  # Odd number of samples
    signal = torch.randn(n_channels, n_samples)
    # Act
    result = ensure_even_len(signal)
    # Assert
    assert torch.equal(result, signal[:, :-1])


def test_ensure_even_len_batch_processing_result_shape_equals_batch_size_n_channels_998():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    batch_size = 32
    n_channels = 16
    n_samples = 999  # Odd number of samples
    signals = torch.randn(batch_size, n_channels, n_samples)
    # Act
    result = ensure_even_len(signals)
    # Assert
    assert result.shape == (batch_size, n_channels, 998)


def test_ensure_even_len_batch_processing_result_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    batch_size = 32
    n_channels = 16
    n_samples = 999
    signals = torch.randn(batch_size, n_channels, n_samples)
    # Act
    result = ensure_even_len(signals)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_batch_processing_torch_equal_result_signals_truncated():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    batch_size = 32
    n_channels = 16
    n_samples = 999
    signals = torch.randn(batch_size, n_channels, n_samples)
    # Act
    result = ensure_even_len(signals)
    # Assert
    assert torch.equal(result, signals[:, :, :-1])


def test_ensure_even_len_fft_compatibility_even_signal_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    signal = torch.randn(1023)  # Odd length
    # Act
    even_signal = ensure_even_len(signal)
    # Assert
    assert even_signal.shape[-1] % 2 == 0


def test_ensure_even_len_fft_compatibility_even_signal_shape_is_1022():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    signal = torch.randn(1023)
    # Act
    even_signal = ensure_even_len(signal)
    # Assert
    assert even_signal.shape[-1] == 1022


def test_ensure_even_len_fft_compatibility_fft_result_shape_equals_even_signal_shape():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    signal = torch.randn(1023)
    even_signal = ensure_even_len(signal)
    # Act
    fft_result = torch.fft.fft(even_signal)
    # Assert
    assert fft_result.shape == even_signal.shape


def test_ensure_even_len_memory_view_result_even_is_x_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x_even = torch.randn(100)
    # Act
    result_even = ensure_even_len(x_even)
    # Assert
    assert result_even is x_even  # Should be the same object


def test_ensure_even_len_memory_view_result_odd_is_not_x_odd():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x_odd = torch.randn(101)
    # Act
    result_odd = ensure_even_len(x_odd)
    # Assert
    assert result_odd is not x_odd  # Should be different objects


def test_ensure_even_len_memory_view_result_odd_shares_data_ptr():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x_odd = torch.randn(101)
    # Act
    result_odd = ensure_even_len(x_odd)
    # Assert
    assert result_odd.data_ptr() == x_odd.data_ptr()  # Same underlying data


@pytest.mark.parametrize(
    "shape",
    [
        (101,),  # 1D
        (10, 101),  # 2D
        (5, 8, 101),  # 3D
        (2, 3, 4, 101),  # 4D
        (2, 2, 2, 2, 101),  # 5D
    ],
)
def test_ensure_even_len_different_shapes_result_shape_matches_expected(shape):
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(shape)
    expected_shape = list(shape)
    expected_shape[-1] -= 1
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape == tuple(expected_shape)


@pytest.mark.parametrize(
    "shape",
    [
        (101,),
        (10, 101),
        (5, 8, 101),
        (2, 3, 4, 101),
        (2, 2, 2, 2, 101),
    ],
)
def test_ensure_even_len_different_shapes_result_shape_last_even(shape):
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(shape)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_integration_test_even_signal_shape_last_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    fs = 1000
    t = torch.linspace(0, 2.001, 2001)  # Odd number of samples
    signal = torch.sin(2 * np.pi * 50 * t) + 0.1 * torch.randn(2001)
    # Act
    even_signal = ensure_even_len(signal)
    # Assert
    assert even_signal.shape[-1] % 2 == 0


def test_ensure_even_len_integration_test_fft_result_shape_equals_windowed_signal_shape():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    fs = 1000
    t = torch.linspace(0, 2.001, 2001)
    signal = torch.sin(2 * np.pi * 50 * t) + 0.1 * torch.randn(2001)
    even_signal = ensure_even_len(signal)
    window = torch.hann_window(even_signal.shape[-1])
    windowed_signal = even_signal * window
    # Act
    fft_result = torch.fft.fft(windowed_signal)
    # Assert
    assert fft_result.shape == windowed_signal.shape


def test_ensure_even_len_integration_test_torch_all_torch_isfinite_fft_result():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    fs = 1000
    t = torch.linspace(0, 2.001, 2001)
    signal = torch.sin(2 * np.pi * 50 * t) + 0.1 * torch.randn(2001)
    even_signal = ensure_even_len(signal)
    window = torch.hann_window(even_signal.shape[-1])
    windowed_signal = even_signal * window
    # Act
    fft_result = torch.fft.fft(windowed_signal)
    # Assert
    assert torch.all(torch.isfinite(fft_result))


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/_ensure_even_len.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-10 11:59:49 (ywatanabe)"
#
#
# def ensure_even_len(x):
#     if x.shape[-1] % 2 == 0:
#         return x
#     else:
#         return x[..., :-1]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/_ensure_even_len.py
# --------------------------------------------------------------------------------
