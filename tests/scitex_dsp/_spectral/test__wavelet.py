#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 14:23:18 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/test__wavelet.py

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from scitex.dsp import wavelet


class TestWavelet:
    """Test cases for wavelet transformation functionality."""

    def test_import_callable_wavelet(self):
        """Test that wavelet can be imported."""
        # Arrange
        # Act
        # Assert
        assert callable(wavelet)

    def test_wavelet_basic_numpy_pha_is_np_ndarray(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert isinstance(pha, np.ndarray)

    def test_wavelet_basic_numpy_amp_is_np_ndarray(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert isinstance(amp, np.ndarray)

    def test_wavelet_basic_numpy_freqs_is_np_ndarray(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert isinstance(freqs, np.ndarray)

    def test_wavelet_basic_numpy_pha_shape_0_1(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert pha.shape[0] == 1  # batch size

    def test_wavelet_basic_numpy_pha_shape_1_1(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert pha.shape[1] == 1  # channels

    def test_wavelet_basic_numpy_amp_shape_equals_pha_shape(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert amp.shape == pha.shape

    def test_wavelet_basic_numpy_len_freqs_shape_1(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert len(freqs.shape) >= 1

    def test_wavelet_basic_numpy_np_all_amp_0(self):
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert np.all(amp >= 0)  # Amplitude should be non-negative


    def test_wavelet_basic_torch_pha_is_torch_tensor(self):
        # Arrange
        fs = 256
        t = torch.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = torch.sin(2 * torch.pi * freq * t).reshape(1, 1, -1)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert isinstance(pha, torch.Tensor)

    def test_wavelet_basic_torch_amp_is_torch_tensor(self):
        # Arrange
        fs = 256
        t = torch.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = torch.sin(2 * torch.pi * freq * t).reshape(1, 1, -1)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert isinstance(amp, torch.Tensor)

    def test_wavelet_basic_torch_freqs_is_torch_tensor(self):
        # Arrange
        fs = 256
        t = torch.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = torch.sin(2 * torch.pi * freq * t).reshape(1, 1, -1)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert isinstance(freqs, torch.Tensor)

    def test_wavelet_basic_torch_torch_all_amp_0(self):
        # Arrange
        fs = 256
        t = torch.linspace(0, 2, 2 * fs)
        freq = 10  # Hz
        x = torch.sin(2 * torch.pi * freq * t).reshape(1, 1, -1)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert torch.all(amp >= 0)


    def test_wavelet_multi_channel_pha_shape_0_1(self):
        # Arrange
        fs = 256
        n_channels = 4
        n_samples = 512
        x = np.random.randn(1, n_channels, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert pha.shape[0] == 1

    def test_wavelet_multi_channel_pha_shape_1_n_channels(self):
        # Arrange
        fs = 256
        n_channels = 4
        n_samples = 512
        x = np.random.randn(1, n_channels, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert pha.shape[1] == n_channels

    def test_wavelet_multi_channel_amp_shape_equals_pha_shape(self):
        # Arrange
        fs = 256
        n_channels = 4
        n_samples = 512
        x = np.random.randn(1, n_channels, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert amp.shape == pha.shape


    def test_wavelet_batch_processing_pha_shape_0_batch_size(self):
        # Arrange
        fs = 256
        batch_size = 3
        n_samples = 512
        x = np.random.randn(batch_size, 2, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu", batch_size=2)
        # Act
        # Assert
        assert pha.shape[0] == batch_size

    def test_wavelet_batch_processing_amp_shape_0_batch_size(self):
        # Arrange
        fs = 256
        batch_size = 3
        n_samples = 512
        x = np.random.randn(batch_size, 2, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu", batch_size=2)
        # Act
        # Assert
        assert amp.shape[0] == batch_size


    def test_wavelet_freq_scale_linear(self):
        """Test wavelet transform with linear frequency scale."""
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)

        pha, amp, freqs = wavelet(x, fs, freq_scale="linear", device="cpu")

        # Check that frequencies are approximately linearly spaced
        # Act
        freq_diffs = np.diff(freqs.flatten())
        # Assert
        assert np.std(freq_diffs) / np.mean(freq_diffs) < 0.1

    def test_wavelet_out_scale_log_not_np_allclose_amp_lin_amp_log(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        pha_lin, amp_lin, _ = wavelet(x, fs, out_scale="linear", device="cpu")
        pha_log, amp_log, _ = wavelet(x, fs, out_scale="log", device="cpu")
        # Phase should be the same
        # Act
        np.testing.assert_allclose(pha_lin, pha_log, rtol=1e-5)
        # Act
        # Assert
        assert not np.allclose(amp_lin, amp_log)

    def test_wavelet_out_scale_log_not_np_any_np_isnan_amp_log(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        pha_lin, amp_lin, _ = wavelet(x, fs, out_scale="linear", device="cpu")
        pha_log, amp_log, _ = wavelet(x, fs, out_scale="log", device="cpu")
        # Phase should be the same
        # Act
        np.testing.assert_allclose(pha_lin, pha_log, rtol=1e-5)
        # Act
        # Assert
        assert not np.any(np.isnan(amp_log))


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wavelet_cuda_device_pha_is_cuda(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = torch.randn(1, 2, n_samples)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cuda")
        # Act
        # Assert
        assert pha.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wavelet_cuda_device_amp_is_cuda(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = torch.randn(1, 2, n_samples)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cuda")
        # Act
        # Assert
        assert amp.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wavelet_cuda_device_freqs_is_cuda(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = torch.randn(1, 2, n_samples)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cuda")
        # Act
        # Assert
        assert freqs.is_cuda


    def test_wavelet_frequency_content(self):
        """Test that wavelet detects correct frequency content."""
        # Arrange
        fs = 256
        t = np.linspace(0, 2, 2 * fs)
        freq = 20  # Hz
        x = np.sin(2 * np.pi * freq * t).reshape(1, 1, -1).astype(np.float32)

        pha, amp, freqs = wavelet(x, fs, device="cpu")

        # amp[0, 0] has shape (n_freqs, n_samples). Average over the
        # time axis to get one scalar per frequency, then pick the
        # peak frequency index.
        amp_per_freq = np.asarray(amp[0, 0]).mean(axis=-1)
        # Act
        peak_freq = np.asarray(freqs).flatten()[np.argmax(amp_per_freq)]

        # Should be close to the input frequency
        # Assert
        assert abs(peak_freq - freq) < 10  # Within 10 Hz tolerance

    def test_wavelet_phase_range_np_all_pha_np_pi_0_1(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert np.all(pha >= -np.pi - 0.1)

    def test_wavelet_phase_range_np_all_pha_np_pi_0_1_2(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert np.all(pha <= np.pi + 0.1)


    def test_wavelet_empty_signal_raises(self):
        """Test that empty signal raises error."""
        # Arrange
        fs = 256
        # Act
        x = np.array([]).reshape(1, 1, 0)

        # Assert
        with pytest.raises(Exception):
            wavelet(x, fs, device="cpu")

    def test_wavelet_time_frequency_dimensions_len_pha_shape_3(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert len(pha.shape) >= 3  # batch, channel, time, freq

    def test_wavelet_time_frequency_dimensions_pha_shape_2_1(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert pha.shape[-2] > 1  # time dimension

    def test_wavelet_time_frequency_dimensions_pha_shape_1_1(self):
        # Arrange
        fs = 256
        n_samples = 512
        x = np.random.randn(1, 1, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Act
        # Assert
        assert pha.shape[-1] > 1  # frequency dimension


    def test_wavelet_chirp_signal(self):
        """Test wavelet transform on chirp signal."""
        # Arrange
        fs = 512
        t = np.linspace(0, 2, 2 * fs)
        # Linear chirp from 10 to 100 Hz
        f0, f1 = 10, 100
        chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 2) * t)
        x = chirp.reshape(1, 1, -1).astype(np.float32)

        pha, amp, freqs = wavelet(x, fs, device="cpu")

        # amp[0, 0] has shape (n_freqs, n_samples). Slice along the
        # time axis (last) to compare early vs late spectra.
        amp_data = np.asarray(amp[0, 0])
        n_samples = amp_data.shape[-1]
        early_amp = amp_data[..., : n_samples // 4].mean(axis=-1)
        late_amp = amp_data[..., -n_samples // 4 :].mean(axis=-1)

        # Find peak frequencies (1-D over the freq axis).
        freqs_arr = np.asarray(freqs).flatten()
        early_peak = freqs_arr[np.argmax(early_amp)]
        # Act
        late_peak = freqs_arr[np.argmax(late_amp)]

        # Late peak should be higher frequency than early peak.
        # Assert
        assert late_peak > early_peak

    def test_wavelet_dtype_preservation_pha_f32_dtype_equals_torch_float32(self):
        # Arrange
        fs = 256
        n_samples = 512
        # Test float32
        x_f32 = torch.randn(1, 1, n_samples, dtype=torch.float32)
        # Act
        pha_f32, amp_f32, _ = wavelet(x_f32, fs, device="cpu")
        # Act
        # Assert
        assert pha_f32.dtype == torch.float32

    def test_wavelet_dtype_preservation_amp_f32_dtype_equals_torch_float32(self):
        # Arrange
        fs = 256
        n_samples = 512
        # Test float32
        x_f32 = torch.randn(1, 1, n_samples, dtype=torch.float32)
        # Act
        pha_f32, amp_f32, _ = wavelet(x_f32, fs, device="cpu")
        # Act
        # Assert
        assert amp_f32.dtype == torch.float32


    def test_wavelet_large_batch_pha_shape_0_batch_size(self):
        # Arrange
        fs = 256
        batch_size = 10
        n_samples = 256
        x = np.random.randn(batch_size, 2, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu", batch_size=4)
        # Act
        # Assert
        assert pha.shape[0] == batch_size

    def test_wavelet_large_batch_amp_shape_0_batch_size(self):
        # Arrange
        fs = 256
        batch_size = 10
        n_samples = 256
        x = np.random.randn(batch_size, 2, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu", batch_size=4)
        # Act
        # Assert
        assert amp.shape[0] == batch_size

    def test_wavelet_large_batch_not_np_any_np_isnan_pha(self):
        # Arrange
        fs = 256
        batch_size = 10
        n_samples = 256
        x = np.random.randn(batch_size, 2, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu", batch_size=4)
        # Act
        # Assert
        assert not np.any(np.isnan(pha))

    def test_wavelet_large_batch_not_np_any_np_isnan_amp(self):
        # Arrange
        fs = 256
        batch_size = 10
        n_samples = 256
        x = np.random.randn(batch_size, 2, n_samples).astype(np.float32)
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu", batch_size=4)
        # Act
        # Assert
        assert not np.any(np.isnan(amp))



if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_wavelet.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:12:00 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/_wavelet.py
#
# """scitex.dsp.wavelet function"""
#
# from scitex.decorators import batch_fn, signal_fn
# from scitex.nn._Wavelet import Wavelet
# import scitex
#
#
# # Functions
# @signal_fn
# @batch_fn
# def wavelet(
#     x,
#     fs,
#     freq_scale="linear",
#     out_scale="linear",
#     device="cuda",
#     batch_size=32,
# ):
#     m = Wavelet(fs, freq_scale=freq_scale, out_scale="linear").to(device).eval()
#     pha, amp, freqs = m(x.to(device))
#
#     if out_scale == "log":
#         amp = (amp + 1e-5).log()
#         if amp.isnan().any():
#             print("NaN is detected while taking the lograrithm of amplitude.")
#
#     return pha, amp, freqs
#
#
# # @signal_fn
# # def wavelet(
# #     x,
# #     fs,
# #     freq_scale="linear",
# #     out_scale="linear",
# #     device="cuda",
# #     batch_size=32,
# # ):
# #     @signal_fn
# #     def _wavelet(
# #         x,
# #         fs,
# #         freq_scale="linear",
# #         out_scale="linear",
# #         device="cuda",
# #     ):
# #         m = (
# #             Wavelet(fs, freq_scale=freq_scale, out_scale=out_scale)
# #             .to(device)
# #             .eval()
# #         )
# #         pha, amp, freqs = m(x.to(device))
#
# #         if out_scale == "log":
# #             amp = (amp + 1e-5).log()
# #             if amp.isnan().any():
# #                 print(
# #                     "NaN is detected while taking the lograrithm of amplitude."
# #                 )
#
# #         return pha, amp, freqs
#
# #     if len(x) <= batch_size:
# #         try:
# #             pha, amp, freqs = _wavelet(
# #                 x,
# #                 fs,
# #                 freq_scale=freq_scale,
# #                 out_scale=out_scale,
# #                 device=device,
# #             )
# #             torch.cuda.empty_cache()
# #             return pha, amp, freqs
#
# #         except Exception as e:
# #             print(e)
# #             print("\nTrying Batch Mode...")
#
# #     n_batches = (len(x) + batch_size - 1) // batch_size
# #     device_orig = x.device
# #     pha, amp, freqs = [], [], []
# #     for i_batch in tqdm(range(n_batches)):
# #         start = i_batch * batch_size
# #         end = (i_batch + 1) * batch_size
# #         _pha, _amp, _freqs = _wavelet(
# #             x[start:end],
# #             fs,
# #             freq_scale=freq_scale,
# #             out_scale=out_scale,
# #             device=device,
# #         )
# #         torch.cuda.empty_cache()
# #         # to CPU
# #         pha.append(_pha.cpu())
# #         amp.append(_amp.cpu())
# #         freqs.append(_freqs.cpu())
#
# #     pha = torch.vstack(pha)
# #     amp = torch.vstack(amp)
# #     freqs = freqs[0]
#
# #     try:
# #         pha = pha.to(device_orig)
# #         amp = amp.to(device_orig)
# #         freqs = freqs.to(device_orig)
# #     except Exception as e:
# #         print(
# #             f"\nError occurred while transferring wavelet outputs back to the original device. Proceeding with CPU tensor. \n\n({e})"
# #         )
#
# #     sleep(0.5)
# #     torch.cuda.empty_cache()
# #     return pha, amp, freqs
#
#
# if __name__ == "__main__":
#     import sys
#
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(sys, plt, agg=True)
#
#     # Parameters
#     FS = 512
#     SIG_TYPE = "chirp"
#     T_SEC = 4
#
#     # Demo signal
#     xx, tt, fs = scitex.dsp.demo_sig(
#         batch_size=64,
#         n_chs=19,
#         n_segments=2,
#         t_sec=T_SEC,
#         fs=FS,
#         sig_type=SIG_TYPE,
#     )
#
#     if SIG_TYPE in ["tensorpac", "pac"]:
#         i_segment = 0
#         xx = xx[:, :, i_segment, :]
#
#     # Main
#     pha, amp, freqs = wavelet(xx, fs, device="cuda")
#     freqs = freqs[0, 0]
#
#     # Plots
#     i_batch, i_ch = 0, 0
#     fig, axes = scitex.plt.subplots(nrows=3)
#
#     # # Time vector for x-axis extents
#     # time_extent = [tt.min(), tt.max()]
#
#     # Trace
#     axes[0].plot(tt, xx[i_batch, i_ch], label=SIG_TYPE)
#     axes[0].set_ylabel("Amplitude [?V]")
#     axes[0].legend(loc="upper left")
#     axes[0].set_title("Signal")
#
#     # Amplitude
#     # extent = [time_extent[0], time_extent[1], freqs.min(), freqs.max()]
#     axes[1].imshow2d(
#         np.log(amp[i_batch, i_ch] + 1e-5).T,
#         cbar_label="Log(amplitude [?V]) [a.u.]",
#         aspect="auto",
#         # extent=extent,
#         # origin="lower",
#     )
#     axes[1] = scitex.plt.ax.set_ticks(axes[1], x_ticks=tt, y_ticks=freqs)
#     axes[1].set_ylabel("Frequency [Hz]")
#     axes[1].set_title("Amplitude")
#
#     # Phase
#     axes[2].imshow2d(
#         pha[i_batch, i_ch].T,
#         cbar_label="Phase [rad]",
#         aspect="auto",
#         # extent=extent,
#         # origin="lower",
#     )
#     axes[2] = scitex.plt.ax.set_ticks(axes[2], x_ticks=tt, y_ticks=freqs)
#     axes[2].set_ylabel("Frequency [Hz]")
#     axes[2].set_title("Phase")
#
#     fig.suptitle("Wavelet Transformation")
#     fig.supxlabel("Time [s]")
#
#     for ax in axes:
#         ax = scitex.plt.ax.set_n_ticks(ax)
#         # ax.set_xlim(time_extent[0], time_extent[1])
#
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     scitex.io.save(fig, "wavelet.png")
#
#     # Close
#     scitex.session.close(CONFIG)
#
# # EOF
#
# """
# /home/ywatanabe/proj/entrance/scitex/dsp/_wavelet.py
# """
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/_wavelet.py
# --------------------------------------------------------------------------------
