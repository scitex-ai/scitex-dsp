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
    # Test 1D tensor with even length
    x = torch.randn(100)  # Even length
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert torch.equal(result, x)


def test_ensure_even_len_already_even_result_shape_equals_x_shape():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 1D tensor with even length
    x = torch.randn(100)  # Even length
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape == x.shape


def test_ensure_even_len_already_even_result_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 1D tensor with even length
    x = torch.randn(100)  # Even length
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape[-1] % 2 == 0




def test_ensure_even_len_odd_to_even_result_shape_equals_n_100():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 1D tensor with odd length
    x = torch.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape == (100,)


def test_ensure_even_len_odd_to_even_result_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 1D tensor with odd length
    x = torch.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_odd_to_even_torch_equal_result_x_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 1D tensor with odd length
    x = torch.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert torch.equal(result, x[:-1])




def test_ensure_even_len_2d_tensor_result_shape_equals_n_10_100():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 2D tensor with odd last dimension
    x = torch.randn(10, 101)  # (batch, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape == (10, 100)


def test_ensure_even_len_2d_tensor_result_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 2D tensor with odd last dimension
    x = torch.randn(10, 101)  # (batch, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_2d_tensor_torch_equal_result_x_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 2D tensor with odd last dimension
    x = torch.randn(10, 101)  # (batch, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert torch.equal(result, x[:, :-1])




def test_ensure_even_len_3d_tensor_result_shape_equals_n_5_8_98():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 3D tensor with odd last dimension
    x = torch.randn(5, 8, 99)  # (batch, channels, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape == (5, 8, 98)


def test_ensure_even_len_3d_tensor_result_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 3D tensor with odd last dimension
    x = torch.randn(5, 8, 99)  # (batch, channels, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_3d_tensor_torch_equal_result_x_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 3D tensor with odd last dimension
    x = torch.randn(5, 8, 99)  # (batch, channels, odd_seq_len)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert torch.equal(result, x[:, :, :-1])




def test_ensure_even_len_4d_tensor_result_shape_equals_n_2_3_4_50():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 4D tensor with odd last dimension
    x = torch.randn(2, 3, 4, 51)  # Odd last dimension
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape == (2, 3, 4, 50)


def test_ensure_even_len_4d_tensor_result_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 4D tensor with odd last dimension
    x = torch.randn(2, 3, 4, 51)  # Odd last dimension
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_4d_tensor_torch_equal_result_x_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test 4D tensor with odd last dimension
    x = torch.randn(2, 3, 4, 51)  # Odd last dimension
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert torch.equal(result, x[:, :, :, :-1])




def test_ensure_even_len_numpy_array_result_shape_equals_n_100():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test with numpy array
    x = np.random.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape == (100,)


def test_ensure_even_len_numpy_array_result_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test with numpy array
    x = np.random.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_numpy_array_np_array_equal_result_x_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test with numpy array
    x = np.random.randn(101)  # Odd length
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert np.array_equal(result, x[:-1])




def test_ensure_even_len_preserves_dtype():
    """Test that ensure_even_len preserves data type."""
    # Arrange
    # Act
    # Assert
    from scitex.dsp.utils import ensure_even_len

    # Test different dtypes
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

    for dtype in dtypes:
        x = torch.randn(101).to(dtype)
        result = ensure_even_len(x)
        assert result.dtype == dtype


def test_ensure_even_len_preserves_device():
    """Test that ensure_even_len preserves device."""
    # Arrange
    from scitex.dsp.utils import ensure_even_len

    # Test CPU tensor
    x_cpu = torch.randn(101)
    # Act
    result_cpu = ensure_even_len(x_cpu)
    # Assert
    assert result_cpu.device == x_cpu.device

    # Test CUDA tensor if available
    if torch.cuda.is_available():
        x_cuda = torch.randn(101).cuda()
        result_cuda = ensure_even_len(x_cuda)
        assert result_cuda.device == x_cuda.device


def test_ensure_even_len_preserves_requires_grad_result_requires_grad_equals_x_requires_grad():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test with requires_grad=True
    x = torch.randn(101, requires_grad=True)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.requires_grad == x.requires_grad


def test_ensure_even_len_preserves_requires_grad_result_requires_grad_equals_x_requires_grad_result_requires_grad_equals_x_requires_grad():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test with requires_grad=True
    x = torch.randn(101, requires_grad=True)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.requires_grad == x.requires_grad


def test_ensure_even_len_preserves_requires_grad_result_requires_grad_equals_x_requires_grad_result_requires_grad_equals_x_requires_grad_2():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test with requires_grad=True
    x = torch.randn(101, requires_grad=True)
    # Act
    result = ensure_even_len(x)
    # Assert
    assert result.requires_grad == x.requires_grad
    # Test with requires_grad=False
    x = torch.randn(101, requires_grad=False)
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.requires_grad == x.requires_grad






def test_ensure_even_len_gradient_flow_x_grad_is_not_none():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101, requires_grad=True)
    result = ensure_even_len(x)
    # Compute a loss and backpropagate
    loss = result.sum()
    # Act
    loss.backward()
    # Act
    # Assert
    assert x.grad is not None


def test_ensure_even_len_gradient_flow_torch_equal_x_grad_100_torch_ones_100():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101, requires_grad=True)
    result = ensure_even_len(x)
    # Compute a loss and backpropagate
    loss = result.sum()
    # Act
    loss.backward()
    # Act
    # Assert
    assert torch.equal(x.grad[:100], torch.ones(100))


def test_ensure_even_len_gradient_flow_x_grad_100_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.randn(101, requires_grad=True)
    result = ensure_even_len(x)
    # Compute a loss and backpropagate
    loss = result.sum()
    # Act
    loss.backward()
    # Act
    # Assert
    assert x.grad[100] == 0  # Last element should have zero gradient




def test_ensure_even_len_small_tensors_result1_shape_equals_n_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test length 1 (odd)
    x1 = torch.tensor([5.0])
    # Act
    result1 = ensure_even_len(x1)
    # Act
    # Assert
    assert result1.shape == (0,)  # Should become empty


def test_ensure_even_len_small_tensors_torch_equal_result2_x2_result1_shape_equals_n_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test length 1 (odd)
    x1 = torch.tensor([5.0])
    # Act
    result1 = ensure_even_len(x1)
    # Act
    # Assert
    assert result1.shape == (0,)  # Should become empty


def test_ensure_even_len_small_tensors_torch_equal_result2_x2_torch_equal_result2_x2():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test length 1 (odd)
    x1 = torch.tensor([5.0])
    # Act
    result1 = ensure_even_len(x1)
    # Assert
    assert result1.shape == (0,)  # Should become empty
    # Test length 2 (even)
    x2 = torch.tensor([1.0, 2.0])
    result2 = ensure_even_len(x2)
    # Act
    # Assert
    assert torch.equal(result2, x2)  # Should remain unchanged




def test_ensure_even_len_small_tensors_torch_equal_result3_torch_tensor_1_0_2_0_result1_shape_equals_n_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test length 1 (odd)
    x1 = torch.tensor([5.0])
    # Act
    result1 = ensure_even_len(x1)
    # Act
    # Assert
    assert result1.shape == (0,)  # Should become empty


def test_ensure_even_len_small_tensors_torch_equal_result3_torch_tensor_1_0_2_0_torch_equal_result2_x2():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test length 1 (odd)
    x1 = torch.tensor([5.0])
    # Act
    result1 = ensure_even_len(x1)
    # Assert
    assert result1.shape == (0,)  # Should become empty
    # Test length 2 (even)
    x2 = torch.tensor([1.0, 2.0])
    result2 = ensure_even_len(x2)
    # Act
    # Assert
    assert torch.equal(result2, x2)  # Should remain unchanged


def test_ensure_even_len_small_tensors_torch_equal_result3_torch_tensor_1_0_2_0_torch_equal_result3_torch_tensor_1_0_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Test length 1 (odd)
    x1 = torch.tensor([5.0])
    # Act
    result1 = ensure_even_len(x1)
    # Assert
    assert result1.shape == (0,)  # Should become empty
    # Test length 2 (even)
    x2 = torch.tensor([1.0, 2.0])
    result2 = ensure_even_len(x2)
    assert torch.equal(result2, x2)  # Should remain unchanged
    # Test length 3 (odd)
    x3 = torch.tensor([1.0, 2.0, 3.0])
    result3 = ensure_even_len(x3)
    # Act
    # Assert
    assert torch.equal(result3, torch.tensor([1.0, 2.0]))






def test_ensure_even_len_empty_tensor_torch_equal_result_x():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.empty(0)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert torch.equal(result, x)


def test_ensure_even_len_empty_tensor_result_shape_equals_n_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    x = torch.empty(0)
    # Act
    result = ensure_even_len(x)
    # Act
    # Assert
    assert result.shape == (0,)




def test_ensure_even_len_consistency():
    """Test that ensure_even_len is consistent across calls."""
    # Arrange
    from scitex.dsp.utils import ensure_even_len

    x = torch.randn(101)

    # Multiple calls should produce identical results
    result1 = ensure_even_len(x)
    # Act
    result2 = ensure_even_len(x)

    # Assert
    assert torch.equal(result1, result2)


def test_ensure_even_len_real_signal_example_signal_shape_1_2_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate EEG signal with odd number of samples. fs * 2.001 = 500.25
    # truncates to 500 (even); use fs * 2.004 = 501.0 → 501 (odd).
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    # Create a sine wave signal
    t = torch.linspace(0, duration, n_samples)
    # Act
    signal = torch.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    # Act
    # Assert
    assert signal.shape[-1] % 2 == 1  # Should be odd


def test_ensure_even_len_real_signal_example_result_shape_1_2_0_signal_shape_1_2_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate EEG signal with odd number of samples. fs * 2.001 = 500.25
    # truncates to 500 (even); use fs * 2.004 = 501.0 → 501 (odd).
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    # Create a sine wave signal
    t = torch.linspace(0, duration, n_samples)
    # Act
    signal = torch.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    # Act
    # Assert
    assert signal.shape[-1] % 2 == 1  # Should be odd


def test_ensure_even_len_real_signal_example_result_shape_1_2_0_result_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate EEG signal with odd number of samples. fs * 2.001 = 500.25
    # truncates to 500 (even); use fs * 2.004 = 501.0 → 501 (odd).
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    # Create a sine wave signal
    t = torch.linspace(0, duration, n_samples)
    # Act
    signal = torch.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    # Assert
    assert signal.shape[-1] % 2 == 1  # Should be odd
    result = ensure_even_len(signal)
    # Act
    # Assert
    assert result.shape[-1] % 2 == 0




def test_ensure_even_len_real_signal_example_result_shape_1_500_signal_shape_1_2_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate EEG signal with odd number of samples. fs * 2.001 = 500.25
    # truncates to 500 (even); use fs * 2.004 = 501.0 → 501 (odd).
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    # Create a sine wave signal
    t = torch.linspace(0, duration, n_samples)
    # Act
    signal = torch.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    # Act
    # Assert
    assert signal.shape[-1] % 2 == 1  # Should be odd


def test_ensure_even_len_real_signal_example_result_shape_1_500_result_shape_1_500():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate EEG signal with odd number of samples. fs * 2.001 = 500.25
    # truncates to 500 (even); use fs * 2.004 = 501.0 → 501 (odd).
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    # Create a sine wave signal
    t = torch.linspace(0, duration, n_samples)
    # Act
    signal = torch.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    # Assert
    assert signal.shape[-1] % 2 == 1  # Should be odd
    result = ensure_even_len(signal)
    # Act
    # Assert
    assert result.shape[-1] == 500  # One less than original




def test_ensure_even_len_real_signal_example_torch_allclose_result_signal_1_signal_shape_1_2_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate EEG signal with odd number of samples. fs * 2.001 = 500.25
    # truncates to 500 (even); use fs * 2.004 = 501.0 → 501 (odd).
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    # Create a sine wave signal
    t = torch.linspace(0, duration, n_samples)
    # Act
    signal = torch.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    # Act
    # Assert
    assert signal.shape[-1] % 2 == 1  # Should be odd


def test_ensure_even_len_real_signal_example_torch_allclose_result_signal_1_torch_allclose_result_signal_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate EEG signal with odd number of samples. fs * 2.001 = 500.25
    # truncates to 500 (even); use fs * 2.004 = 501.0 → 501 (odd).
    fs = 250  # Hz
    duration = 2.004
    n_samples = int(fs * duration)  # 501 (odd)
    # Create a sine wave signal
    t = torch.linspace(0, duration, n_samples)
    # Act
    signal = torch.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    # Assert
    assert signal.shape[-1] % 2 == 1  # Should be odd
    result = ensure_even_len(signal)
    # Act
    # Assert
    assert torch.allclose(result, signal[:-1])






def test_ensure_even_len_multichannel_signal_result_shape_equals_n_channels_1000():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate multi-channel EEG data
    n_channels = 64
    n_samples = 1001  # Odd number of samples
    signal = torch.randn(n_channels, n_samples)
    # Act
    result = ensure_even_len(signal)
    # Act
    # Assert
    assert result.shape == (n_channels, 1000)


def test_ensure_even_len_multichannel_signal_result_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate multi-channel EEG data
    n_channels = 64
    n_samples = 1001  # Odd number of samples
    signal = torch.randn(n_channels, n_samples)
    # Act
    result = ensure_even_len(signal)
    # Act
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_multichannel_signal_torch_equal_result_signal_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate multi-channel EEG data
    n_channels = 64
    n_samples = 1001  # Odd number of samples
    signal = torch.randn(n_channels, n_samples)
    # Act
    result = ensure_even_len(signal)
    # Act
    # Assert
    assert torch.equal(result, signal[:, :-1])




def test_ensure_even_len_batch_processing_result_shape_equals_batch_size_n_channels_998():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate batch of signals
    batch_size = 32
    n_channels = 16
    n_samples = 999  # Odd number of samples
    signals = torch.randn(batch_size, n_channels, n_samples)
    # Act
    result = ensure_even_len(signals)
    # Act
    # Assert
    assert result.shape == (batch_size, n_channels, 998)


def test_ensure_even_len_batch_processing_result_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate batch of signals
    batch_size = 32
    n_channels = 16
    n_samples = 999  # Odd number of samples
    signals = torch.randn(batch_size, n_channels, n_samples)
    # Act
    result = ensure_even_len(signals)
    # Act
    # Assert
    assert result.shape[-1] % 2 == 0


def test_ensure_even_len_batch_processing_torch_equal_result_signals_1():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate batch of signals
    batch_size = 32
    n_channels = 16
    n_samples = 999  # Odd number of samples
    signals = torch.randn(batch_size, n_channels, n_samples)
    # Act
    result = ensure_even_len(signals)
    # Act
    # Assert
    assert torch.equal(result, signals[:, :, :-1])




def test_ensure_even_len_fft_compatibility_even_signal_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Create a signal that needs to be even for FFT
    signal = torch.randn(1023)  # Odd length
    # Ensure even length for FFT
    # Act
    even_signal = ensure_even_len(signal)
    # Act
    # Assert
    assert even_signal.shape[-1] % 2 == 0


def test_ensure_even_len_fft_compatibility_even_signal_shape_1_1022():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Create a signal that needs to be even for FFT
    signal = torch.randn(1023)  # Odd length
    # Ensure even length for FFT
    # Act
    even_signal = ensure_even_len(signal)
    # Act
    # Assert
    assert even_signal.shape[-1] == 1022


def test_ensure_even_len_fft_compatibility_fft_result_shape_equals_even_signal_shape_even_signal_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Create a signal that needs to be even for FFT
    signal = torch.randn(1023)  # Odd length
    # Ensure even length for FFT
    # Act
    even_signal = ensure_even_len(signal)
    # Act
    # Assert
    assert even_signal.shape[-1] % 2 == 0


def test_ensure_even_len_fft_compatibility_fft_result_shape_equals_even_signal_shape_even_signal_shape_1_1022():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Create a signal that needs to be even for FFT
    signal = torch.randn(1023)  # Odd length
    # Ensure even length for FFT
    # Act
    even_signal = ensure_even_len(signal)
    # Act
    # Assert
    assert even_signal.shape[-1] == 1022


def test_ensure_even_len_fft_compatibility_fft_result_shape_equals_even_signal_shape_fft_result_shape_equals_even_signal_shape():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Create a signal that needs to be even for FFT
    signal = torch.randn(1023)  # Odd length
    # Ensure even length for FFT
    # Act
    even_signal = ensure_even_len(signal)
    # Should now be compatible with FFT requirements
    # Assert
    assert even_signal.shape[-1] % 2 == 0
    assert even_signal.shape[-1] == 1022
    # Should be able to compute FFT without issues
    fft_result = torch.fft.fft(even_signal)
    # Act
    # Assert
    assert fft_result.shape == even_signal.shape






def test_ensure_even_len_memory_view_result_even_is_x_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # For even length, should return the same tensor (view)
    x_even = torch.randn(100)
    # Act
    result_even = ensure_even_len(x_even)
    # Act
    # Assert
    assert result_even is x_even  # Should be the same object


def test_ensure_even_len_memory_view_result_odd_is_not_x_odd_result_even_is_x_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # For even length, should return the same tensor (view)
    x_even = torch.randn(100)
    # Act
    result_even = ensure_even_len(x_even)
    # Act
    # Assert
    assert result_even is x_even  # Should be the same object


def test_ensure_even_len_memory_view_result_odd_is_not_x_odd_result_odd_is_not_x_odd():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # For even length, should return the same tensor (view)
    x_even = torch.randn(100)
    # Act
    result_even = ensure_even_len(x_even)
    # Assert
    assert result_even is x_even  # Should be the same object
    # For odd length, should return a new view (slice)
    x_odd = torch.randn(101)
    result_odd = ensure_even_len(x_odd)
    # Act
    # Assert
    assert result_odd is not x_odd  # Should be different objects




def test_ensure_even_len_memory_view_result_odd_data_ptr_x_odd_data_ptr_result_even_is_x_even():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # For even length, should return the same tensor (view)
    x_even = torch.randn(100)
    # Act
    result_even = ensure_even_len(x_even)
    # Act
    # Assert
    assert result_even is x_even  # Should be the same object


def test_ensure_even_len_memory_view_result_odd_data_ptr_x_odd_data_ptr_result_odd_data_ptr_x_odd_data_ptr():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # For even length, should return the same tensor (view)
    x_even = torch.randn(100)
    # Act
    result_even = ensure_even_len(x_even)
    # Assert
    assert result_even is x_even  # Should be the same object
    # For odd length, should return a new view (slice)
    x_odd = torch.randn(101)
    result_odd = ensure_even_len(x_odd)
    # Act
    # Assert
    assert result_odd.data_ptr() == x_odd.data_ptr()  # But same underlying data






def test_ensure_even_len_different_shapes():
    """Test ensure_even_len with various tensor shapes."""
    # Arrange
    # Act
    # Assert
    from scitex.dsp.utils import ensure_even_len

    # Test various shapes where last dimension is odd
    shapes = [
        (101,),  # 1D
        (10, 101),  # 2D
        (5, 8, 101),  # 3D
        (2, 3, 4, 101),  # 4D
        (2, 2, 2, 2, 101),  # 5D
    ]

    for shape in shapes:
        x = torch.randn(shape)
        result = ensure_even_len(x)

        # Last dimension should be reduced by 1
        expected_shape = list(shape)
        expected_shape[-1] -= 1

        assert result.shape == tuple(expected_shape)
        assert result.shape[-1] % 2 == 0


def test_ensure_even_len_integration_test_even_signal_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate a complete signal processing pipeline
    # 1. Generate raw signal with odd length
    fs = 1000
    t = torch.linspace(0, 2.001, 2001)  # Odd number of samples
    signal = torch.sin(2 * np.pi * 50 * t) + 0.1 * torch.randn(2001)
    # 2. Ensure even length for downstream processing
    # Act
    even_signal = ensure_even_len(signal)
    # Act
    # Assert
    assert even_signal.shape[-1] % 2 == 0


def test_ensure_even_len_integration_test_fft_result_shape_equals_windowed_signal_shape_even_signal_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate a complete signal processing pipeline
    # 1. Generate raw signal with odd length
    fs = 1000
    t = torch.linspace(0, 2.001, 2001)  # Odd number of samples
    signal = torch.sin(2 * np.pi * 50 * t) + 0.1 * torch.randn(2001)
    # 2. Ensure even length for downstream processing
    # Act
    even_signal = ensure_even_len(signal)
    # Act
    # Assert
    assert even_signal.shape[-1] % 2 == 0


def test_ensure_even_len_integration_test_fft_result_shape_equals_windowed_signal_shape_fft_result_shape_equals_windowed_signal_shape():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate a complete signal processing pipeline
    # 1. Generate raw signal with odd length
    fs = 1000
    t = torch.linspace(0, 2.001, 2001)  # Odd number of samples
    signal = torch.sin(2 * np.pi * 50 * t) + 0.1 * torch.randn(2001)
    # 2. Ensure even length for downstream processing
    # Act
    even_signal = ensure_even_len(signal)
    # Assert
    assert even_signal.shape[-1] % 2 == 0
    # 3. Apply windowing (requires even length)
    window = torch.hann_window(even_signal.shape[-1])
    windowed_signal = even_signal * window
    # 4. Compute FFT (works better with even lengths)
    fft_result = torch.fft.fft(windowed_signal)
    # Act
    # Assert
    assert fft_result.shape == windowed_signal.shape




def test_ensure_even_len_integration_test_torch_all_torch_isfinite_fft_result_even_signal_shape_1_2_0():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate a complete signal processing pipeline
    # 1. Generate raw signal with odd length
    fs = 1000
    t = torch.linspace(0, 2.001, 2001)  # Odd number of samples
    signal = torch.sin(2 * np.pi * 50 * t) + 0.1 * torch.randn(2001)
    # 2. Ensure even length for downstream processing
    # Act
    even_signal = ensure_even_len(signal)
    # Act
    # Assert
    assert even_signal.shape[-1] % 2 == 0


def test_ensure_even_len_integration_test_torch_all_torch_isfinite_fft_result_torch_all_torch_isfinite_fft_result():
    # Arrange
    from scitex.dsp.utils import ensure_even_len
    # Simulate a complete signal processing pipeline
    # 1. Generate raw signal with odd length
    fs = 1000
    t = torch.linspace(0, 2.001, 2001)  # Odd number of samples
    signal = torch.sin(2 * np.pi * 50 * t) + 0.1 * torch.randn(2001)
    # 2. Ensure even length for downstream processing
    # Act
    even_signal = ensure_even_len(signal)
    # Assert
    assert even_signal.shape[-1] % 2 == 0
    # 3. Apply windowing (requires even length)
    window = torch.hann_window(even_signal.shape[-1])
    windowed_signal = even_signal * window
    # 4. Compute FFT (works better with even lengths)
    fft_result = torch.fft.fft(windowed_signal)
    # Act
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
