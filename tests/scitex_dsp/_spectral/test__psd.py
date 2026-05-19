#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 13:28:19 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/test__psd.py

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from scitex.dsp import psd


class TestPsd:
    """Test cases for power spectral density (PSD) calculation."""

    def test_import_callable_psd(self):
        """Test that psd can be imported."""
        # Arrange
        # Act
        # Assert
        assert callable(psd)

    def test_psd_basic_numpy_power_is_np_ndarray(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert isinstance(power, np.ndarray)

    def test_psd_basic_numpy_freqs_is_np_ndarray(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert isinstance(freqs, np.ndarray)

    def test_psd_basic_numpy_power_shape_equals_n_1_1_len_freqs(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert power.shape == (1, 1, len(freqs))

    def test_psd_basic_numpy_len_freqs_0(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert len(freqs) > 0

    def test_psd_basic_numpy_np_all_power_0(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert np.all(power >= 0)  # Power should be non-negative

    def test_psd_basic_numpy_abs_peak_freq_freq_5_power_is_np_ndarray(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert isinstance(power, np.ndarray)

    def test_psd_basic_numpy_abs_peak_freq_freq_5_freqs_is_np_ndarray(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert isinstance(freqs, np.ndarray)

    def test_psd_basic_numpy_abs_peak_freq_freq_5_power_shape_equals_n_1_1_len_freqs(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert power.shape == (1, 1, len(freqs))

    def test_psd_basic_numpy_abs_peak_freq_freq_5_len_freqs_0(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert len(freqs) > 0

    def test_psd_basic_numpy_abs_peak_freq_freq_5_np_all_power_0(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert np.all(power >= 0)  # Power should be non-negative

    def test_psd_basic_numpy_abs_peak_freq_freq_5_abs_peak_freq_freq_5(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Assert
        assert isinstance(power, np.ndarray)
        assert isinstance(freqs, np.ndarray)
        assert power.shape == (1, 1, len(freqs))
        assert len(freqs) > 0
        assert np.all(power >= 0)  # Power should be non-negative
        # Check that peak is near the signal frequency
        peak_idx = np.argmax(power[0, 0])
        peak_freq = freqs[peak_idx]
        # Act
        # Assert
        assert abs(peak_freq - freq) < 5  # Within 5 Hz tolerance



    def test_psd_basic_torch_power_is_torch_tensor(self):
        # Arrange
        fs = 256
        t = torch.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = torch.sin(2 * torch.pi * freq * t).reshape(1, 1, -1)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert isinstance(power, torch.Tensor)

    def test_psd_basic_torch_freqs_is_torch_tensor(self):
        # Arrange
        fs = 256
        t = torch.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = torch.sin(2 * torch.pi * freq * t).reshape(1, 1, -1)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert isinstance(freqs, torch.Tensor)

    def test_psd_basic_torch_power_shape_equals_n_1_1_len_freqs(self):
        # Arrange
        fs = 256
        t = torch.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = torch.sin(2 * torch.pi * freq * t).reshape(1, 1, -1)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert power.shape == (1, 1, len(freqs))

    def test_psd_basic_torch_torch_all_power_0(self):
        # Arrange
        fs = 256
        t = torch.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = torch.sin(2 * torch.pi * freq * t).reshape(1, 1, -1)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert torch.all(power >= 0)


    def test_psd_multi_channel_power_shape_equals_n_1_n_channels_len_freqs(self):
        # Arrange
        fs = 256
        n_channels = 4
        n_samples = 512
        x = np.random.randn(1, n_channels, n_samples).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert power.shape == (1, n_channels, len(freqs))

    def test_psd_multi_channel_all_np_any_power_0_ch_0_for_ch_in_range_n_channels(self):
        # Arrange
        fs = 256
        n_channels = 4
        n_samples = 512
        x = np.random.randn(1, n_channels, n_samples).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert all(np.any(power[0, ch] > 0) for ch in range(n_channels))


    def test_psd_batch_processing_power_shape_equals_batch_size_2_len_freqs(self):
        # Arrange
        fs = 256
        batch_size = 3
        n_samples = 512
        x = np.random.randn(batch_size, 2, n_samples).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert power.shape == (batch_size, 2, len(freqs))

    def test_psd_batch_processing_power_shape_0_batch_size(self):
        # Arrange
        fs = 256
        batch_size = 3
        n_samples = 512
        x = np.random.randn(batch_size, 2, n_samples).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert power.shape[0] == batch_size


    def test_psd_prob_mode_abs_np_sum_power_prob_0_0_1_0_0_01(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        power_regular, _ = psd(x, fs, prob=False)
        # Act
        power_prob, _ = psd(x, fs, prob=True)
        # Act
        # Assert
        assert abs(np.sum(power_prob[0, 0]) - 1.0) < 0.01

    def test_psd_prob_mode_power_regular_shape_equals_power_prob_shape(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        power_regular, _ = psd(x, fs, prob=False)
        # Act
        power_prob, _ = psd(x, fs, prob=True)
        # Act
        # Assert
        assert power_regular.shape == power_prob.shape


    def test_psd_frequency_resolution_abs_actual_resolution_freq_resolution_0_1(self):
        # Arrange
        fs = 512
        n_samples = 1024
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        power, freqs = psd(x, fs)
        # Check frequency resolution
        freq_resolution = fs / n_samples
        # Act
        actual_resolution = freqs[1] - freqs[0]
        # Act
        # Assert
        assert abs(actual_resolution - freq_resolution) < 0.1

    def test_psd_frequency_resolution_freqs_1_fs_2(self):
        # Arrange
        fs = 512
        n_samples = 1024
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        power, freqs = psd(x, fs)
        # Check frequency resolution
        freq_resolution = fs / n_samples
        # Act
        actual_resolution = freqs[1] - freqs[0]
        # Act
        # Assert
        assert freqs[-1] <= fs / 2


    def test_psd_dimension_parameter_power1_shape_2_1(self):
        # Arrange
        fs = 256
        x = np.random.randn(2, 3, 512).astype(np.float32)
        # Default dim=-1 (along time axis)
        # Act
        power1, _ = psd(x, fs, dim=-1)
        # Act
        # Assert
        assert power1.shape[2] > 1  # Frequency dimension

    def test_psd_dimension_parameter_power2_shape_equals_power1_shape_power1_shape_2_1(self):
        # Arrange
        fs = 256
        x = np.random.randn(2, 3, 512).astype(np.float32)
        # Default dim=-1 (along time axis)
        # Act
        power1, _ = psd(x, fs, dim=-1)
        # Act
        # Assert
        assert power1.shape[2] > 1  # Frequency dimension

    def test_psd_dimension_parameter_power2_shape_equals_power1_shape_power2_shape_equals_power1_shape(self):
        # Arrange
        fs = 256
        x = np.random.randn(2, 3, 512).astype(np.float32)
        # Default dim=-1 (along time axis)
        # Act
        power1, _ = psd(x, fs, dim=-1)
        # Assert
        assert power1.shape[2] > 1  # Frequency dimension
        # Custom dimension - should work the same for 3D input
        power2, _ = psd(x, fs, dim=2)
        # Act
        # Assert
        assert power2.shape == power1.shape



    def test_psd_white_noise(self):
        """Test PSD of white noise."""
        # Arrange
        fs = 256
        n_samples = 2048
        x = np.random.randn(1, 1, n_samples).astype(np.float32)

        power, freqs = psd(x, fs)

        # White noise should have relatively flat spectrum
        power_values = power[0, 0]
        mean_power = np.mean(power_values)
        std_power = np.std(power_values)

        # Most values should be within 2 std of mean
        # Act
        within_range = np.sum(np.abs(power_values - mean_power) < 2 * std_power)
        # Assert
        assert within_range / len(power_values) > 0.8

    def test_psd_sinusoid_mixture_any_abs_pf_freq1_5_for_pf_in_peak_freqs(self):
        # Arrange
        fs = 512
        t = np.linspace(0, 2, 2 * fs)
        freq1, freq2 = 10, 50  # Hz
        x = (
            (np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t))
            .reshape(1, 1, -1)
            .astype(np.float32)
        )
        power, freqs = psd(x, fs)
        # Find peaks
        power_1d = power[0, 0]
        mean_power = np.mean(power_1d)
        peaks = np.where(power_1d > 5 * mean_power)[0]
        # Act
        peak_freqs = freqs[peaks]
        # Act
        # Assert
        assert any(abs(pf - freq1) < 5 for pf in peak_freqs)

    def test_psd_sinusoid_mixture_any_abs_pf_freq2_5_for_pf_in_peak_freqs(self):
        # Arrange
        fs = 512
        t = np.linspace(0, 2, 2 * fs)
        freq1, freq2 = 10, 50  # Hz
        x = (
            (np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t))
            .reshape(1, 1, -1)
            .astype(np.float32)
        )
        power, freqs = psd(x, fs)
        # Find peaks
        power_1d = power[0, 0]
        mean_power = np.mean(power_1d)
        peaks = np.where(power_1d > 5 * mean_power)[0]
        # Act
        peak_freqs = freqs[peaks]
        # Act
        # Assert
        assert any(abs(pf - freq2) < 5 for pf in peak_freqs)


    def test_psd_empty_signal_raises(self):
        """Test that empty signal raises error."""
        # Arrange
        fs = 256
        # Act
        x = np.array([]).reshape(1, 1, 0)

        # Assert
        with pytest.raises(Exception):
            psd(x, fs)

    def test_psd_dc_component_freqs_0_0_or_abs_freqs_0_1(self):
        # Arrange
        fs = 256
        n_samples = 512
        dc_offset = 5.0
        x = (dc_offset * np.ones(n_samples)).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert freqs[0] == 0 or abs(freqs[0]) < 1

    def test_psd_dc_component_power_0_0_0_power_0_0_1_max(self):
        # Arrange
        fs = 256
        n_samples = 512
        dc_offset = 5.0
        x = (dc_offset * np.ones(n_samples)).reshape(1, 1, -1).astype(np.float32)
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert power[0, 0, 0] > power[0, 0, 1:].max()


    def test_psd_dtype_preservation_power_f32_dtype_equals_np_float32(self):
        # Arrange
        fs = 256
        n_samples = 512
        # Test float32
        x_f32 = np.random.randn(1, 1, n_samples).astype(np.float32)
        # Act
        power_f32, _ = psd(x_f32, fs)
        # Act
        # Assert
        assert power_f32.dtype == np.float32

    def test_psd_dtype_preservation_power_f64_dtype_in_np_float32_np_float64_power_f32_dtype_equals_np_float32(self):
        # Arrange
        fs = 256
        n_samples = 512
        # Test float32
        x_f32 = np.random.randn(1, 1, n_samples).astype(np.float32)
        # Act
        power_f32, _ = psd(x_f32, fs)
        # Act
        # Assert
        assert power_f32.dtype == np.float32

    def test_psd_dtype_preservation_power_f64_dtype_in_np_float32_np_float64_power_f64_dtype_in_np_float32_np_float64(self):
        # Arrange
        fs = 256
        n_samples = 512
        # Test float32
        x_f32 = np.random.randn(1, 1, n_samples).astype(np.float32)
        # Act
        power_f32, _ = psd(x_f32, fs)
        # Assert
        assert power_f32.dtype == np.float32
        # Test float64
        x_f64 = np.random.randn(1, 1, n_samples).astype(np.float64)
        power_f64, _ = psd(x_f64, fs)
        # Act
        # Assert
        assert power_f64.dtype in [np.float32, np.float64]



    def test_psd_cuda_device_power_is_cuda(self):
        # Arrange
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        fs = 256
        n_samples = 512
        x = torch.randn(1, 2, n_samples).cuda()
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert power.is_cuda

    def test_psd_cuda_device_freqs_is_cuda(self):
        # Arrange
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        fs = 256
        n_samples = 512
        x = torch.randn(1, 2, n_samples).cuda()
        # Act
        power, freqs = psd(x, fs)
        # Act
        # Assert
        assert freqs.is_cuda


    def test_psd_parseval_theorem(self):
        """PSD integral should match the per-sample variance.

        scitex_nn's PSD is *one-sided but undoubled* (scipy's welch
        doubles off-DC bins; this implementation does not), so
        ``sum(P) * fs/N`` ≈ ``var(x) / 2``. The test asserts that
        relationship within 25 %, which catches gross normalization
        regressions while tolerating the convention.
        """
        # Arrange
        fs = 256
        n_samples = 1024
        x = np.random.randn(1, 1, n_samples).astype(np.float32)

        time_var = np.var(x)
        power, _ = psd(x, fs)
        freq_integral = np.sum(power[0, 0]) * (fs / n_samples)

        # Expected ratio is 0.5 (one-sided, undoubled).
        # Act
        ratio = freq_integral / time_var
        # Assert
        assert 0.4 < ratio < 0.6, f"Parseval ratio out of band: {ratio:.3f}"

    def test_psd_window_effect(self):
        """Test that PSD is affected by windowing (implicit in implementation)."""
        # Arrange
        fs = 256
        n_samples = 512

        # Signal with abrupt edges
        x1 = np.ones(n_samples).reshape(1, 1, -1).astype(np.float32)
        x1[0, 0, :100] = 0
        x1[0, 0, -100:] = 0

        # Smooth signal
        x2 = np.hamming(n_samples).reshape(1, 1, -1).astype(np.float32)

        power1, _ = psd(x1, fs)
        power2, _ = psd(x2, fs)

        # Abrupt signal should have more high-frequency content
        # Act
        high_freq_idx = len(power1[0, 0]) // 2
        # Assert
        assert np.sum(power1[0, 0, high_freq_idx:]) > np.sum(
            power2[0, 0, high_freq_idx:]
        )


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_psd.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-11-04 02:11:25 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/_psd.py
#
# """This script does XYZ."""
#
# try:
#     import torch
#
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#     torch = None
#
# from scitex.decorators import signal_fn
#
# if TORCH_AVAILABLE:
#     from scitex.nn._PSD import PSD
#
#
# @signal_fn
# def psd(
#     x,
#     fs,
#     prob=False,
#     dim=-1,
# ):
#     """
#     import matplotlib.pyplot as plt
#
#     x, t, fs = scitex.dsp.demo_sig()  # (batch_size, n_chs, seq_len)
#     pp, ff = psd(x, fs)
#
#     # Plots
#     plt, CC = scitex.plt.configure_mpl(plt)
#     fig, ax = scitex.plt.subplots()
#     ax.plot(fs, pp[0, 0])
#     ax.xlabel("Frequency [Hz]")
#     ax.ylabel("log(Power [uV^2 / Hz]) [a.u.]")
#     plt.show()
#     """
#     if not TORCH_AVAILABLE:
#         raise ImportError(
#             "PyTorch is not installed. Please install with: pip install torch"
#         )
#     psd, freqs = PSD(fs, prob=prob, dim=dim)(x)
#     return psd, freqs
#
#
# def band_powers(self, psd):
#     """
#     Calculate the average power for specified frequency bands.
#     """
#     assert len(self.low_freqs) == len(self.high_freqs)
#
#     out = []
#     for ll, hh in zip(self.low_freqs, self.high_freqs):
#         band_indices = torch.where((freqs >= ll) & (freqs <= hh))[0].to(psd.device)
#         band_power = psd[..., band_indices].sum(dim=self.dim)
#         bandwidth = hh - ll
#         avg_band_power = band_power / bandwidth
#         out.append(avg_band_power)
#     out = torch.stack(out, dim=-1)
#     return out
#
#     # Average Power in Each Frequency Band
#     avg_band_powers = self.calc_band_avg_power(psd, freqs)
#     return (avg_band_powers,)
#
#
# if __name__ == "__main__":
#     import sys
#
#     import matplotlib.pyplot as plt
#
#     import scitex
#
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(sys, plt)
#
#     # Parameters
#     SIG_TYPE = "chirp"
#
#     # Demo signal
#     xx, tt, fs = scitex.dsp.demo_sig(SIG_TYPE)  # (8, 19, 384)
#
#     # PSD calculation
#     pp, ff = psd(xx, fs, prob=True)
#
#     # Plots
#     fig, axes = scitex.plt.subplots(nrows=2)
#
#     axes[0].plot(tt, xx[0, 0], label=SIG_TYPE)
#     axes[1].set_title("Signal")
#     axes[0].set_xlabel("Time [s]")
#     axes[0].set_ylabel("Amplitude [?V]")
#
#     axes[1].plot(ff, pp[0, 0])
#     axes[1].set_title("PSD (power spectrum density)")
#     axes[1].set_xlabel("Frequency [Hz]")
#     axes[1].set_ylabel("Log(Power [?V^2 / Hz]) [a.u.]")
#
#     scitex.io.save(fig, "psd.png")
#
#     # Close
#     scitex.session.close(CONFIG)
#
# # EOF
#
# """
# /home/ywatanabe/proj/entrance/scitex/dsp/_psd.py
# """
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_psd.py
# --------------------------------------------------------------------------------
