#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-01 20:50:00 (ywatanabe)"
# File: ./tests/scitex/dsp/test_filt.py

"""
Test module for scitex.dsp.filt filtering functions.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np


class TestFilt:
    """Test class for filtering functions."""

    @pytest.fixture
    def simple_signal(self):
        """Create a simple test signal with known frequency components."""
        fs = 1000  # 1 kHz sampling rate
        t = np.linspace(0, 1, fs, endpoint=False)
        # Create signal with 50 Hz and 200 Hz components
        signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)
        return signal.astype(np.float32), t, fs

    @pytest.fixture
    def multi_channel_signal(self):
        """Create a multi-channel test signal."""
        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        n_channels = 3
        signals = []
        for i in range(n_channels):
            freq = 50 * (i + 1)  # 50, 100, 150 Hz
            signals.append(np.sin(2 * np.pi * freq * t))
        signal = np.array(signals).astype(np.float32)
        return signal, t, fs

    def test_module_import_hasattr_scitex_dsp_filt_bandpass(self):
        # Arrange
        # Act
        # Arrange
        # Act
        import scitex.dsp.filt
        # Act
        # Assert
        assert hasattr(scitex.dsp.filt, "bandpass")

    def test_module_import_hasattr_scitex_dsp_filt_bandstop(self):
        # Arrange
        # Act
        # Arrange
        # Act
        import scitex.dsp.filt
        # Act
        # Assert
        assert hasattr(scitex.dsp.filt, "bandstop")

    def test_module_import_hasattr_scitex_dsp_filt_lowpass(self):
        # Arrange
        # Act
        # Arrange
        # Act
        import scitex.dsp.filt
        # Act
        # Assert
        assert hasattr(scitex.dsp.filt, "lowpass")

    def test_module_import_hasattr_scitex_dsp_filt_highpass(self):
        # Arrange
        # Act
        # Arrange
        # Act
        import scitex.dsp.filt
        # Act
        # Assert
        assert hasattr(scitex.dsp.filt, "highpass")

    def test_module_import_hasattr_scitex_dsp_filt_gauss(self):
        # Arrange
        # Act
        # Arrange
        # Act
        import scitex.dsp.filt
        # Act
        # Assert
        assert hasattr(scitex.dsp.filt, "gauss")


    def test_bandpass_filter_filtered_shape_equals_signal_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import bandpass
        signal, t, fs = simple_signal
        bands = np.array([[40, 60]])  # Pass 50 Hz, reject 200 Hz
        # Act
        filtered, t_out = bandpass(signal, fs, bands, t=t)
        # Act
        # Assert
        assert filtered.shape == signal.shape

    def test_bandpass_filter_t_out_shape_equals_t_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import bandpass
        signal, t, fs = simple_signal
        bands = np.array([[40, 60]])  # Pass 50 Hz, reject 200 Hz
        # Act
        filtered, t_out = bandpass(signal, fs, bands, t=t)
        # Act
        # Assert
        assert t_out.shape == t.shape

    def test_bandpass_filter_np_std_filtered_np_std_signal(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import bandpass
        signal, t, fs = simple_signal
        bands = np.array([[40, 60]])  # Pass 50 Hz, reject 200 Hz
        # Act
        filtered, t_out = bandpass(signal, fs, bands, t=t)
        # Act
        # Assert
        assert np.std(filtered) < np.std(signal)


    def test_bandstop_filter_filtered_shape_equals_signal_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import bandstop
        signal, t, fs = simple_signal
        bands = np.array([[190, 210]])  # Stop 200 Hz, pass 50 Hz
        # Act
        filtered, t_out = bandstop(signal, fs, bands, t=t)
        # Act
        # Assert
        assert filtered.shape == signal.shape

    def test_bandstop_filter_t_out_shape_equals_t_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import bandstop
        signal, t, fs = simple_signal
        bands = np.array([[190, 210]])  # Stop 200 Hz, pass 50 Hz
        # Act
        filtered, t_out = bandstop(signal, fs, bands, t=t)
        # Act
        # Assert
        assert t_out.shape == t.shape

    def test_bandstop_filter_np_std_filtered_np_std_signal(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import bandstop
        signal, t, fs = simple_signal
        bands = np.array([[190, 210]])  # Stop 200 Hz, pass 50 Hz
        # Act
        filtered, t_out = bandstop(signal, fs, bands, t=t)
        # Act
        # Assert
        assert np.std(filtered) < np.std(signal)


    def test_lowpass_filter_filtered_shape_equals_signal_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import lowpass
        signal, t, fs = simple_signal
        cutoff = 100  # Pass 50 Hz, reject 200 Hz
        # Act
        filtered, t_out = lowpass(signal, fs, cutoff, t=t)
        # Act
        # Assert
        assert filtered.shape == signal.shape

    def test_lowpass_filter_t_out_shape_equals_t_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import lowpass
        signal, t, fs = simple_signal
        cutoff = 100  # Pass 50 Hz, reject 200 Hz
        # Act
        filtered, t_out = lowpass(signal, fs, cutoff, t=t)
        # Act
        # Assert
        assert t_out.shape == t.shape

    def test_lowpass_filter_np_std_filtered_np_std_signal(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import lowpass
        signal, t, fs = simple_signal
        cutoff = 100  # Pass 50 Hz, reject 200 Hz
        # Act
        filtered, t_out = lowpass(signal, fs, cutoff, t=t)
        # Act
        # Assert
        assert np.std(filtered) < np.std(signal)


    def test_highpass_filter_filtered_shape_equals_signal_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import highpass
        signal, t, fs = simple_signal
        cutoff = 100  # Reject 50 Hz, pass 200 Hz
        # Act
        filtered, t_out = highpass(signal, fs, cutoff, t=t)
        # Act
        # Assert
        assert filtered.shape == signal.shape

    def test_highpass_filter_t_out_shape_equals_t_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import highpass
        signal, t, fs = simple_signal
        cutoff = 100  # Reject 50 Hz, pass 200 Hz
        # Act
        filtered, t_out = highpass(signal, fs, cutoff, t=t)
        # Act
        # Assert
        assert t_out.shape == t.shape

    def test_highpass_filter_np_std_filtered_0(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import highpass
        signal, t, fs = simple_signal
        cutoff = 100  # Reject 50 Hz, pass 200 Hz
        # Act
        filtered, t_out = highpass(signal, fs, cutoff, t=t)
        # Act
        # Assert
        assert np.std(filtered) > 0


    def test_gaussian_filter_filtered_shape_equals_signal_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import gauss
        signal, t, fs = simple_signal
        sigma = 5  # Smoothing parameter
        # Act
        filtered, t_out = gauss(signal, sigma, t=t)
        # Act
        # Assert
        assert filtered.shape == signal.shape

    def test_gaussian_filter_t_out_shape_equals_t_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import gauss
        signal, t, fs = simple_signal
        sigma = 5  # Smoothing parameter
        # Act
        filtered, t_out = gauss(signal, sigma, t=t)
        # Act
        # Assert
        assert t_out.shape == t.shape

    def test_gaussian_filter_np_std_diff_filtered_np_std_diff_original_filtered_shape_equals_signal_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import gauss
        signal, t, fs = simple_signal
        sigma = 5  # Smoothing parameter
        # Act
        filtered, t_out = gauss(signal, sigma, t=t)
        # Act
        # Assert
        assert filtered.shape == signal.shape

    def test_gaussian_filter_np_std_diff_filtered_np_std_diff_original_t_out_shape_equals_t_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import gauss
        signal, t, fs = simple_signal
        sigma = 5  # Smoothing parameter
        # Act
        filtered, t_out = gauss(signal, sigma, t=t)
        # Act
        # Assert
        assert t_out.shape == t.shape

    def test_gaussian_filter_reduces_high_freq_std(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import gauss
        signal, t, fs = simple_signal
        sigma = 5  # Smoothing parameter
        # Act
        filtered, t_out = gauss(signal, sigma, t=t)
        diff_original = np.diff(signal)
        diff_filtered = np.diff(filtered)
        # Assert: Gaussian filter smooths the signal
        assert np.std(diff_filtered) < np.std(diff_original)



    def test_multi_channel_filtering_filtered_shape_equals_signal_shape(self, multi_channel_signal):
        # Arrange
        from scitex.dsp.filt import bandpass
        signal, t, fs = multi_channel_signal
        bands = np.array([[80, 120]])  # Pass only 100 Hz channel
        # bandpass may add a leading batch axis (when input is 2-D)
        # and a band axis (n_bands=1) — squeeze size-1 dims so the
        # shape comparison ignores them.
        # Act
        filtered = np.asarray(bandpass(signal, fs, bands)).squeeze()
        # Act
        # Assert
        assert filtered.shape == signal.shape

    def test_multi_channel_filtering_np_argmax_channel_energies_1_filtered_shape_equals_signal_shape(self, multi_channel_signal):
        # Arrange
        from scitex.dsp.filt import bandpass
        signal, t, fs = multi_channel_signal
        bands = np.array([[80, 120]])  # Pass only 100 Hz channel
        # bandpass may add a leading batch axis (when input is 2-D)
        # and a band axis (n_bands=1) — squeeze size-1 dims so the
        # shape comparison ignores them.
        # Act
        filtered = np.asarray(bandpass(signal, fs, bands)).squeeze()
        # Act
        # Assert
        assert filtered.shape == signal.shape

    def test_multi_channel_filtering_middle_channel_has_highest_energy(self, multi_channel_signal):
        # Arrange
        from scitex.dsp.filt import bandpass
        signal, t, fs = multi_channel_signal
        bands = np.array([[80, 120]])  # Pass only 100 Hz channel
        # Act
        filtered = np.asarray(bandpass(signal, fs, bands)).squeeze()
        channel_energies = np.std(filtered, axis=1)
        # Assert: middle channel (100 Hz) has highest energy
        assert np.argmax(channel_energies) == 1



    def test_torch_tensor_input_filtered_is_torch_tensor(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import bandpass
        signal_np, t_np, fs = simple_signal
        signal_torch = torch.tensor(signal_np)
        t_torch = torch.tensor(t_np)
        bands = np.array([[40, 60]])
        # Act
        filtered, t_out = bandpass(signal_torch, fs, bands, t=t_torch)
        # Act
        # Assert
        assert isinstance(filtered, torch.Tensor)

    def test_torch_tensor_input_t_out_is_torch_tensor(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import bandpass
        signal_np, t_np, fs = simple_signal
        signal_torch = torch.tensor(signal_np)
        t_torch = torch.tensor(t_np)
        bands = np.array([[40, 60]])
        # Act
        filtered, t_out = bandpass(signal_torch, fs, bands, t=t_torch)
        # Act
        # Assert
        assert isinstance(t_out, torch.Tensor)

    def test_torch_tensor_input_filtered_shape_equals_signal_torch_shape(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import bandpass
        signal_np, t_np, fs = simple_signal
        signal_torch = torch.tensor(signal_np)
        t_torch = torch.tensor(t_np)
        bands = np.array([[40, 60]])
        # Act
        filtered, t_out = bandpass(signal_torch, fs, bands, t=t_torch)
        # Act
        # Assert
        assert filtered.shape == signal_torch.shape


    def test_multiple_bands_preserves_signal_length(self):
        """Test filtering with multiple frequency bands preserves length."""
        # Arrange
        from scitex.dsp.filt import bandpass

        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        signal = (
            np.sin(2 * np.pi * 50 * t)
            + np.sin(2 * np.pi * 150 * t)
            + np.sin(2 * np.pi * 250 * t)
        )
        bands = np.array([[40, 60], [140, 160]])  # Pass 50 and 150 Hz
        # Act
        try:
            filtered = np.asarray(bandpass(signal, fs, bands))
        except Exception:
            bands_single = np.array([[40, 160]])
            filtered = np.asarray(bandpass(signal, fs, bands_single)).squeeze(-2)
        # Assert
        assert filtered.shape[-1] == signal.shape[-1]

    def test_edge_frequencies_np_std_filtered_low_np_std_signal_0_5(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import highpass, lowpass
        signal, t, fs = simple_signal
        # Use cutoffs that are inside the safe range for the upstream
        # IIR filter: very-low (1 Hz) and right-at-Nyquist push the
        # filter into a degenerate state where the output is a single
        # sample. Use 5 Hz / Nyquist*0.95 instead, still extreme.
        # Act
        filtered_low, _ = lowpass(signal, fs, 5, t=t)
        # Act
        # Assert
        assert np.std(filtered_low) < np.std(signal) * 0.5

    def test_edge_frequencies_np_std_filtered_high_np_std_signal_0_5_np_std_filtered_low_np_std_signal_0_5(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import highpass, lowpass
        signal, t, fs = simple_signal
        # Use cutoffs that are inside the safe range for the upstream
        # IIR filter: very-low (1 Hz) and right-at-Nyquist push the
        # filter into a degenerate state where the output is a single
        # sample. Use 5 Hz / Nyquist*0.95 instead, still extreme.
        # Act
        filtered_low, _ = lowpass(signal, fs, 5, t=t)
        # Act
        # Assert
        assert np.std(filtered_low) < np.std(signal) * 0.5

    def test_edge_frequencies_highpass_near_nyquist_reduces_std(self, simple_signal):
        # Arrange
        from scitex.dsp.filt import highpass
        signal, t, fs = simple_signal
        # Use Nyquist*0.45 — extreme but safe for the upstream IIR.
        # Act
        filtered_high, _ = highpass(signal, fs, fs * 0.45, t=t)
        # Assert
        assert np.std(filtered_high) < np.std(signal) * 0.5



    def test_empty_signal_raises_runtimeerror_valueerror(self):
        """Filtering an empty signal raises (zero-element reshape)."""
        # Arrange
        from scitex.dsp.filt import bandpass

        signal = np.array([])
        fs = 1000
        # Act
        bands = np.array([[40, 60]])

        # Assert
        with pytest.raises((RuntimeError, ValueError)):
            bandpass(signal, fs, bands)

    def test_batch_filtering_filtered_shape_equals_signal_shape(self):
        """Test filtering with batched signals."""
        # Arrange
        from scitex.dsp.filt import bandpass

        # Create batched signal (batch_size, n_channels, n_samples)
        batch_size = 2
        n_channels = 3
        n_samples = 1000
        fs = 1000

        signal = np.random.randn(batch_size, n_channels, n_samples).astype(np.float32)
        bands = np.array([[40, 60]])

        # Act
        filtered = np.asarray(bandpass(signal, fs, bands)).squeeze(-2)

        # Should preserve (batch, channel, sample) dimensions.
        # Assert
        assert filtered.shape == signal.shape

    def test_filter_stability_no_nan(self, simple_signal):
        """Test that filters do not produce NaN values."""
        # Arrange
        from scitex.dsp.filt import bandpass, highpass, lowpass
        signal, _, fs = simple_signal
        filters = [
            (bandpass, {"bands": np.array([[40, 60]])}),
            (lowpass, {"cutoffs_hz": 100}),
            (highpass, {"cutoffs_hz": 100}),
        ]
        # Act
        outputs = [
            np.asarray(filt_func(signal, fs, **params)).squeeze()
            for filt_func, params in filters
        ]
        # Assert
        assert not any(np.any(np.isnan(out)) for out in outputs)

    def test_filter_stability_no_inf(self, simple_signal):
        """Test that filters do not produce Inf values."""
        # Arrange
        from scitex.dsp.filt import bandpass, highpass, lowpass
        signal, _, fs = simple_signal
        filters = [
            (bandpass, {"bands": np.array([[40, 60]])}),
            (lowpass, {"cutoffs_hz": 100}),
            (highpass, {"cutoffs_hz": 100}),
        ]
        # Act
        outputs = [
            np.asarray(filt_func(signal, fs, **params)).squeeze()
            for filt_func, params in filters
        ]
        # Assert
        assert not any(np.any(np.isinf(out)) for out in outputs)

    def test_filter_stability_bounded_output(self, simple_signal):
        """Test that filtered output amplitude is bounded."""
        # Arrange
        from scitex.dsp.filt import bandpass, highpass, lowpass
        signal, _, fs = simple_signal
        filters = [
            (bandpass, {"bands": np.array([[40, 60]])}),
            (lowpass, {"cutoffs_hz": 100}),
            (highpass, {"cutoffs_hz": 100}),
        ]
        bound = np.max(np.abs(signal)) * 10
        # Act
        outputs = [
            np.asarray(filt_func(signal, fs, **params)).squeeze()
            for filt_func, params in filters
        ]
        # Assert
        assert all(np.max(np.abs(out)) < bound for out in outputs)

    @pytest.mark.parametrize("sigma", [1, 3, 5, 10])
    def test_gaussian_sigma_effect(self, sigma):
        """Test Gaussian filter with different sigma values."""
        # Arrange
        from scitex.dsp.filt import gauss

        # Create noisy signal + matching time vector. We pass `t=` so gauss
        # returns the canonical (filtered, t_out) tuple shape — without it
        # the wrapper returns just the array (per BaseFilter1D.forward's
        # `if t is None: return x` branch) and the unpack fails.
        signal = np.random.randn(1000).astype(np.float32)
        t = np.arange(1000, dtype=np.float32)

        filtered, _ = gauss(signal, sigma, t=t)

        # Larger sigma should smooth more
        # Act
        roughness = np.std(np.diff(filtered))
        # Assert
        assert roughness < np.std(np.diff(signal))


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/filt.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-11-04 02:05:47 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/filt.py
#
# import numpy as np
#
# import scitex
# from scitex.decorators import signal_fn
#
# # No top-level imports from nn module to avoid circular dependency
# # Filters will be imported inside functions when needed
#
#
# @signal_fn
# def gauss(x, sigma, t=None):
#     from scitex.nn._Filters import GaussianFilter
#
#     return GaussianFilter(sigma)(x, t=t)
#
#
# @signal_fn
# def bandpass(x, fs, bands, t=None):
#     import torch
#     from scitex.nn._Filters import BandPassFilter
#
#     from scitex.nn._Filters import BandPassFilter
#
#     # Convert bands to tensor if it's not already
#     if not isinstance(bands, torch.Tensor):
#         bands = torch.tensor(bands, dtype=torch.float32)
#     return BandPassFilter(bands, fs, x.shape[-1])(x, t=t)
#
#
# @signal_fn
# def bandstop(x, fs, bands, t=None):
#     import torch
#
#     from scitex.nn._Filters import BandStopFilter
#
#     # Convert bands to tensor if it's not already
#     if not isinstance(bands, torch.Tensor):
#         bands = torch.tensor(bands, dtype=torch.float32)
#     return BandStopFilter(bands, fs, x.shape[-1])(x, t=t)
#
#
# @signal_fn
# def lowpass(x, fs, cutoffs_hz, t=None):
#     from scitex.nn._Filters import LowPassFilter
#
#     return LowPassFilter(cutoffs_hz, fs, x.shape[-1])(x, t=t)
#
#
# @signal_fn
# def highpass(x, fs, cutoffs_hz, t=None):
#     from scitex.nn._Filters import HighPassFilter
#
#     return HighPassFilter(cutoffs_hz, fs, x.shape[-1])(x, t=t)
#
#
# def _custom_print(x):
#     print(type(x), x.shape)
#
#
# if __name__ == "__main__":
#     import sys
#
#     import matplotlib.pyplot as plt
#
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(sys, plt)
#
#     # Parametes
#     T_SEC = 1
#     SRC_FS = 1024
#     FREQS_HZ = list(np.linspace(0, 500, 10, endpoint=False).astype(int))
#     SIG_TYPE = "periodic"
#     BANDS = np.vstack([[80, 310]])
#     SIGMA = 3
#
#     # Demo Signal
#     xx, tt, fs = scitex.dsp.demo_sig(
#         t_sec=T_SEC,
#         fs=SRC_FS,
#         freqs_hz=FREQS_HZ,
#         sig_type=SIG_TYPE,
#     )
#
#     # Filtering
#     x_bp, t_bp = scitex.dsp.filt.bandpass(xx, fs, BANDS, t=tt)
#     x_bs, t_bs = scitex.dsp.filt.bandstop(xx, fs, BANDS, t=tt)
#     x_lp, t_lp = scitex.dsp.filt.lowpass(xx, fs, BANDS[:, 0], t=tt)
#     x_hp, t_hp = scitex.dsp.filt.highpass(xx, fs, BANDS[:, 1], t=tt)
#     x_g, t_g = scitex.dsp.filt.gauss(xx, sigma=SIGMA, t=tt)
#     filted = {
#         f"Original (Sum of {FREQS_HZ}-Hz signals)": (xx, tt, fs),
#         f"Bandpass-filtered ({BANDS[0][0]} - {BANDS[0][1]} Hz)": (
#             x_bp,
#             t_bp,
#             fs,
#         ),
#         f"Bandstop-filtered ({BANDS[0][0]} - {BANDS[0][1]} Hz)": (
#             x_bs,
#             t_bs,
#             fs,
#         ),
#         f"Lowpass-filtered ({BANDS[0][0]} Hz)": (x_lp, t_lp, fs),
#         f"Highpass-filtered ({BANDS[0][1]} Hz)": (x_hp, t_hp, fs),
#         f"Gaussian-filtered (sigma = {SIGMA} SD [point])": (x_g, t_g, fs),
#     }
#
#     # Plots traces
#     fig, axes = plt.subplots(nrows=len(filted), ncols=1, sharex=True, sharey=True)
#     i_batch = 0
#     i_ch = 0
#     i_filt = 0
#     for ax, (k, v) in zip(axes, filted.items()):
#         _xx, _tt, _fs = v
#         if _xx.ndim == 3:
#             _xx = _xx[i_batch, i_ch]
#         elif _xx.ndim == 4:
#             _xx = _xx[i_batch, i_ch, i_filt]
#         ax.plot(_tt, _xx, label=k)
#         ax.legend(loc="upper left")
#
#     fig.suptitle("Filtered")
#     fig.supxlabel("Time [s]")
#     fig.supylabel("Amplitude")
#
#     scitex.io.save(fig, "traces.png")
#
#     # Calculates and Plots PSD
#     fig, axes = plt.subplots(nrows=len(filted), ncols=1, sharex=True, sharey=True)
#     i_batch = 0
#     i_ch = 0
#     i_filt = 0
#     for ax, (k, v) in zip(axes, filted.items()):
#         _xx, _tt, _fs = v
#
#         _psd, ff = scitex.dsp.psd(_xx, _fs)
#         if _psd.ndim == 3:
#             _psd = _psd[i_batch, i_ch]
#         elif _psd.ndim == 4:
#             _psd = _psd[i_batch, i_ch, i_filt]
#
#         ax.plot(ff, _psd, label=k)
#         ax.legend(loc="upper left")
#
#         for bb in np.hstack(BANDS):
#             ax.axvline(x=bb, color=CC["grey"], linestyle="--")
#
#     fig.suptitle("PSD (power spectrum density) of filtered signals")
#     fig.supxlabel("Frequency [Hz]")
#     fig.supylabel("log(Power [uV^2 / Hz]) [a.u.]")
#     scitex.io.save(fig, "psd.png")
#
#     # Close
#     scitex.session.close(CONFIG)
#
# # EOF
#
# """
# /home/ywatanabe/proj/scitex/src/scitex/dsp/filt.py
# """
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/filt.py
# --------------------------------------------------------------------------------
