#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 10:52:57 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/test__hilbert.py

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from scitex.dsp import hilbert


class TestHilbert:
    """Test cases for hilbert transform function."""

    @pytest.fixture
    def simple_signal(self):
        """Create a simple sinusoidal signal."""
        t = np.linspace(0, 1, 1000)
        freq = 10  # Hz
        signal = np.sin(2 * np.pi * freq * t)
        return signal.astype(np.float32)

    @pytest.fixture
    def complex_signal(self):
        """Create a multi-frequency signal."""
        t = np.linspace(0, 1, 1000)
        signal = (
            np.sin(2 * np.pi * 5 * t)
            + 0.5 * np.sin(2 * np.pi * 20 * t)
            + 0.3 * np.sin(2 * np.pi * 50 * t)
        )
        return signal.astype(np.float32)

    def test_import_callable_hilbert(self):
        """Test that hilbert can be imported."""
        # Arrange
        # Act
        # Assert
        assert callable(hilbert)

    def test_numpy_1d_signal_phase_is_np_ndarray(self, simple_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert isinstance(phase, np.ndarray)

    def test_numpy_1d_signal_amplitude_is_np_ndarray(self, simple_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert isinstance(amplitude, np.ndarray)

    def test_numpy_1d_signal_phase_shape_equals_simple_signal_shape(self, simple_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert phase.shape == simple_signal.shape

    def test_numpy_1d_signal_amplitude_shape_equals_simple_signal_shape(self, simple_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert amplitude.shape == simple_signal.shape

    def test_numpy_1d_signal_phase_dtype_equals_np_float32(self, simple_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert phase.dtype == np.float32

    def test_numpy_1d_signal_amplitude_dtype_equals_np_float32(self, simple_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert amplitude.dtype == np.float32


    def test_torch_1d_signal_phase_is_torch_tensor(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal)
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Act
        # Assert
        assert isinstance(phase, torch.Tensor)

    def test_torch_1d_signal_amplitude_is_torch_tensor(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal)
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Act
        # Assert
        assert isinstance(amplitude, torch.Tensor)

    def test_torch_1d_signal_phase_shape_equals_tensor_signal_shape(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal)
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Act
        # Assert
        assert phase.shape == tensor_signal.shape

    def test_torch_1d_signal_amplitude_shape_equals_tensor_signal_shape(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal)
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Act
        # Assert
        assert amplitude.shape == tensor_signal.shape

    def test_torch_1d_signal_phase_dtype_equals_torch_float32(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal)
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Act
        # Assert
        assert phase.dtype == torch.float32

    def test_torch_1d_signal_amplitude_dtype_equals_torch_float32(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal)
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Act
        # Assert
        assert amplitude.dtype == torch.float32


    def test_2d_signal_phase_shape_equals_signal_2d_shape(self, simple_signal):
        # Arrange
        signal_2d = np.stack([simple_signal] * 4)
        # Act
        phase, amplitude = hilbert(signal_2d)
        # Act
        # Assert
        assert phase.shape == signal_2d.shape

    def test_2d_signal_amplitude_shape_equals_signal_2d_shape(self, simple_signal):
        # Arrange
        signal_2d = np.stack([simple_signal] * 4)
        # Act
        phase, amplitude = hilbert(signal_2d)
        # Act
        # Assert
        assert amplitude.shape == signal_2d.shape


    def test_3d_signal_phase_shape_equals_signal_3d_shape(self, simple_signal):
        # Arrange
        signal_3d = np.stack([np.stack([simple_signal] * 4)] * 2)
        # Act
        phase, amplitude = hilbert(signal_3d)
        # Act
        # Assert
        assert phase.shape == signal_3d.shape

    def test_3d_signal_amplitude_shape_equals_signal_3d_shape(self, simple_signal):
        # Arrange
        signal_3d = np.stack([np.stack([simple_signal] * 4)] * 2)
        # Act
        phase, amplitude = hilbert(signal_3d)
        # Act
        # Assert
        assert amplitude.shape == signal_3d.shape


    def test_phase_amplitude_relationship_np_all_amplitude_0(self, simple_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert np.all(amplitude >= 0)

    def test_phase_amplitude_relationship_np_all_phase_np_pi(self, simple_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert np.all(phase >= -np.pi)

    def test_phase_amplitude_relationship_np_all_phase_np_pi_2(self, simple_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert np.all(phase <= np.pi)

    def test_phase_amplitude_relationship_np_std_center_amp_np_mean_center_amp_0_01_np_all_amplitude_0(self, simple_signal):
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert np.all(amplitude >= 0)

    def test_phase_amplitude_relationship_np_std_center_amp_np_mean_center_amp_0_01_np_all_phase_np_pi(self, simple_signal):
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert np.all(phase >= -np.pi)

    def test_phase_amplitude_relationship_np_std_center_amp_np_mean_center_amp_0_01_np_all_phase_np_pi_2(self, simple_signal):
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Act
        # Assert
        assert np.all(phase <= np.pi)

    def test_phase_amplitude_relationship_center_amplitude_near_constant(self, simple_signal):
        """Sine wave: center-window amplitude std/mean < 1%."""
        # Arrange
        # Act
        phase, amplitude = hilbert(simple_signal)
        # Assert
        center_amp = amplitude[100:-100]
        assert np.std(center_amp) / np.mean(center_amp) < 0.01



    def test_constant_signal_np_allclose_amplitude_100_100_1_0_rtol_0_1(self):
        """Test hilbert transform on constant signal."""
        # Arrange
        constant = np.ones(1000, dtype=np.float32)
        # Act
        phase, amplitude = hilbert(constant)

        # Constant signal should have near-constant amplitude
        # Phase is undefined for DC component
        # Assert
        assert np.allclose(amplitude[100:-100], 1.0, rtol=0.1)

    def test_zero_signal_np_allclose_amplitude_0_0_atol_1e_06(self):
        """Test hilbert transform on zero signal."""
        # Arrange
        zeros = np.zeros(1000, dtype=np.float32)
        # Act
        phase, amplitude = hilbert(zeros)

        # Zero signal should have zero amplitude
        # Assert
        assert np.allclose(amplitude, 0.0, atol=1e-6)

    def test_complex_signal_frequencies_amplitude_min_0(self, complex_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(complex_signal)
        # Act
        # Assert
        assert amplitude.min() >= 0

    def test_complex_signal_frequencies_amplitude_max_2_0(self, complex_signal):
        # Arrange
        # Act
        # Arrange
        # Act
        phase, amplitude = hilbert(complex_signal)
        # Act
        # Assert
        assert amplitude.max() <= 2.0  # Max possible for our complex signal


    def test_dim_parameter_phase1_shape_equals_signal_2d_shape(self, simple_signal):
        # Arrange
        signal_2d = np.stack([simple_signal, simple_signal * 0.5])
        # Transform along last dimension (default)
        # Act
        phase1, amp1 = hilbert(signal_2d, dim=-1)
        # Act
        # Assert
        assert phase1.shape == signal_2d.shape

    def test_dim_parameter_phase2_shape_equals_signal_2d_shape_phase1_shape_equals_signal_2d_shape(self, simple_signal):
        # Arrange
        signal_2d = np.stack([simple_signal, simple_signal * 0.5])
        # Transform along last dimension (default)
        # Act
        phase1, amp1 = hilbert(signal_2d, dim=-1)
        # Act
        # Assert
        assert phase1.shape == signal_2d.shape

    def test_dim_parameter_dim0_phase_shape_preserved(self, simple_signal):
        """Hilbert along dim=0 preserves signal shape."""
        # Arrange
        signal_2d = np.stack([simple_signal, simple_signal * 0.5])
        # Act
        phase2, amp2 = hilbert(signal_2d, dim=0)
        # Assert
        assert phase2.shape == signal_2d.shape


    def test_dim_parameter_not_np_allclose_phase1_phase2_phase1_shape_equals_signal_2d_shape(self, simple_signal):
        # Arrange
        signal_2d = np.stack([simple_signal, simple_signal * 0.5])
        # Transform along last dimension (default)
        # Act
        phase1, amp1 = hilbert(signal_2d, dim=-1)
        # Act
        # Assert
        assert phase1.shape == signal_2d.shape

    def test_dim_parameter_phase_differs_between_dims(self, simple_signal):
        """Phase along dim=-1 differs from phase along dim=0."""
        # Arrange
        signal_2d = np.stack([simple_signal, simple_signal * 0.5])
        phase1, _ = hilbert(signal_2d, dim=-1)
        # Act
        phase2, _ = hilbert(signal_2d, dim=0)
        # Assert
        assert not np.allclose(phase1, phase2)


    def test_dim_parameter_not_np_allclose_amp1_amp2_phase1_shape_equals_signal_2d_shape(self, simple_signal):
        # Arrange
        signal_2d = np.stack([simple_signal, simple_signal * 0.5])
        # Transform along last dimension (default)
        # Act
        phase1, amp1 = hilbert(signal_2d, dim=-1)
        # Act
        # Assert
        assert phase1.shape == signal_2d.shape

    def test_dim_parameter_amplitude_differs_between_dims(self, simple_signal):
        """Amplitude along dim=-1 differs from amplitude along dim=0."""
        # Arrange
        signal_2d = np.stack([simple_signal, simple_signal * 0.5])
        _, amp1 = hilbert(signal_2d, dim=-1)
        # Act
        _, amp2 = hilbert(signal_2d, dim=0)
        # Assert
        assert not np.allclose(amp1, amp2)



    def test_instantaneous_frequency_np_abs_np_mean_center_freq_10_0_0_5(self, simple_signal):
        """Test that instantaneous frequency can be derived from phase."""
        # Arrange
        phase, amplitude = hilbert(simple_signal)

        # Compute instantaneous frequency from phase derivative
        fs = 1000  # Sampling frequency
        inst_freq = np.diff(np.unwrap(phase)) * fs / (2 * np.pi)

        # For a 10 Hz sine wave, instantaneous frequency should be ~10 Hz
        # (except at boundaries)
        # Act
        center_freq = inst_freq[100:-100]
        # Assert
        assert np.abs(np.mean(center_freq) - 10.0) < 0.5

    def test_analytic_signal_property(self, simple_signal):
        """Test that hilbert transform creates proper analytic signal."""
        # Arrange
        phase, amplitude = hilbert(simple_signal)

        # Reconstruct analytic signal
        analytic = amplitude * np.exp(1j * phase)

        # Real part should approximate original signal
        reconstructed = np.real(analytic)

        # Allow some error at boundaries
        center_orig = simple_signal[50:-50]
        center_recon = reconstructed[50:-50]
        # Act
        correlation = np.corrcoef(center_orig, center_recon)[0, 1]
        # Assert
        assert correlation > 0.99

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_device_handling_phase_device_equals_tensor_signal_device(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal).cuda()
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Assert
        assert phase.device == tensor_signal.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_device_handling_amplitude_device_equals_tensor_signal_device(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal).cuda()
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Assert
        assert amplitude.device == tensor_signal.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_device_handling_phase_is_cuda(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal).cuda()
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Assert
        assert phase.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_device_handling_amplitude_is_cuda(self, simple_signal):
        # Arrange
        tensor_signal = torch.from_numpy(simple_signal).cuda()
        # Act
        phase, amplitude = hilbert(tensor_signal)
        # Assert
        assert amplitude.is_cuda


    def test_batch_consistency_np_allclose_phase1_phase2_0_rtol_1e_05(self, simple_signal):
        # Arrange
        phase1, amp1 = hilbert(simple_signal)
        # Batched signal
        batched = np.stack([simple_signal, simple_signal])
        # Act
        phase2, amp2 = hilbert(batched)
        # Act
        # Assert
        assert np.allclose(phase1, phase2[0], rtol=1e-5)

    def test_batch_consistency_np_allclose_phase1_phase2_1_rtol_1e_05(self, simple_signal):
        # Arrange
        phase1, amp1 = hilbert(simple_signal)
        # Batched signal
        batched = np.stack([simple_signal, simple_signal])
        # Act
        phase2, amp2 = hilbert(batched)
        # Act
        # Assert
        assert np.allclose(phase1, phase2[1], rtol=1e-5)

    def test_batch_consistency_np_allclose_amp1_amp2_0_rtol_1e_05(self, simple_signal):
        # Arrange
        phase1, amp1 = hilbert(simple_signal)
        # Batched signal
        batched = np.stack([simple_signal, simple_signal])
        # Act
        phase2, amp2 = hilbert(batched)
        # Act
        # Assert
        assert np.allclose(amp1, amp2[0], rtol=1e-5)

    def test_batch_consistency_np_allclose_amp1_amp2_1_rtol_1e_05(self, simple_signal):
        # Arrange
        phase1, amp1 = hilbert(simple_signal)
        # Batched signal
        batched = np.stack([simple_signal, simple_signal])
        # Act
        phase2, amp2 = hilbert(batched)
        # Act
        # Assert
        assert np.allclose(amp1, amp2[1], rtol=1e-5)


    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_preservation_phase(self, dtype):
        """hilbert preserves dtype on phase output."""
        # Arrange
        signal = np.random.randn(1000).astype(dtype)
        # Act
        phase, _ = hilbert(signal)
        # Assert
        assert phase.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_preservation_amplitude(self, dtype):
        """hilbert preserves dtype on amplitude output."""
        # Arrange
        signal = np.random.randn(1000).astype(dtype)
        # Act
        _, amplitude = hilbert(signal)
        # Assert
        assert amplitude.dtype == dtype

    def test_empty_signal_phase_shape_equals_n_0(self):
        # Arrange
        empty = np.array([], dtype=np.float32)
        # Act
        phase, amplitude = hilbert(empty)
        # Act
        # Assert
        assert phase.shape == (0,)

    def test_empty_signal_amplitude_shape_equals_n_0(self):
        # Arrange
        empty = np.array([], dtype=np.float32)
        # Act
        phase, amplitude = hilbert(empty)
        # Act
        # Assert
        assert amplitude.shape == (0,)


    def test_single_sample_phase_shape_equals_n_1(self):
        # Arrange
        single = np.array([1.0], dtype=np.float32)
        # Act
        phase, amplitude = hilbert(single)
        # Act
        # Assert
        assert phase.shape == (1,)

    def test_single_sample_amplitude_shape_equals_n_1(self):
        # Arrange
        single = np.array([1.0], dtype=np.float32)
        # Act
        phase, amplitude = hilbert(single)
        # Act
        # Assert
        assert amplitude.shape == (1,)

    def test_single_sample_np_isfinite_phase_0(self):
        # Arrange
        single = np.array([1.0], dtype=np.float32)
        # Act
        phase, amplitude = hilbert(single)
        # Act
        # Assert
        assert np.isfinite(phase[0])

    def test_single_sample_np_isfinite_amplitude_0(self):
        # Arrange
        single = np.array([1.0], dtype=np.float32)
        # Act
        phase, amplitude = hilbert(single)
        # Act
        # Assert
        assert np.isfinite(amplitude[0])



if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_hilbert.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:07:11 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/_hilbert.py
#
# """
# This script does XYZ.
# """
#
# import sys
#
# import matplotlib.pyplot as plt
# from scitex.nn._Hilbert import Hilbert
#
# from scitex.decorators import signal_fn
#
#
# # Functions
# @signal_fn
# def hilbert(
#     x,
#     dim=-1,
# ):
#     y = Hilbert(x.shape[-1], dim=dim)(x)
#     return y[..., 0], y[..., 1]
#
#
# if __name__ == "__main__":
#     import scitex
#
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(sys, plt)
#
#     # Parameters
#     T_SEC = 1.0
#     FS = 400
#     SIG_TYPE = "chirp"
#
#     # Demo signal
#     xx, tt, fs = scitex.dsp.demo_sig(t_sec=T_SEC, fs=FS, sig_type=SIG_TYPE)
#
#     # Main
#     pha, amp = hilbert(
#         xx,
#         dim=-1,
#     )
#     # (32, 19, 1280, 2)
#
#     # Plots
#     fig, axes = scitex.plt.subplots(nrows=2, sharex=True)
#     fig.suptitle("Hilbert Transformation")
#
#     axes[0].plot(tt, xx[0, 0], label=SIG_TYPE)
#     axes[0].plot(tt, amp[0, 0], label="Amplidue")
#     axes[0].legend()
#     # axes[0].set_xlabel("Time [s]")
#     axes[0].set_ylabel("Amplitude [?V]")
#
#     axes[1].plot(tt, pha[0, 0], label="Phase")
#     axes[1].legend()
#
#     axes[1].set_xlabel("Time [s]")
#     axes[1].set_ylabel("Phase [rad]")
#
#     # plt.show()
#     scitex.io.save(fig, "traces.png")
#
#     # Close
#     scitex.session.close(CONFIG)
#
# # EOF
#
# """
# /home/ywatanabe/proj/entrance/scitex/dsp/_hilbert.py
# """
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_hilbert.py
# --------------------------------------------------------------------------------
