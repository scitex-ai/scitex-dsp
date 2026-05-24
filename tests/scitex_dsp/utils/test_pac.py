#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:24:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/utils/test_pac.py

"""Tests for PAC (Phase-Amplitude Coupling) functionality."""

import pytest


def test_three_pac_public_surfaces_resolve_callable_scitex_dsp_pac():
    # Arrange
    pytest.importorskip("tensorpac")
    import scitex_dsp
    # Act
    from scitex_dsp.utils import _calc_pac_with_tensorpac, pac
    # Act
    # Assert
    assert callable(scitex_dsp.pac)


def test_three_pac_public_surfaces_resolve_hasattr_pac_calc_pac_with_tensorpac():
    # Arrange
    pytest.importorskip("tensorpac")
    import scitex_dsp
    # Act
    from scitex_dsp.utils import _calc_pac_with_tensorpac, pac
    # Act
    # Assert
    assert hasattr(pac, "calc_pac_with_tensorpac")


def test_three_pac_public_surfaces_resolve_calc_pac_with_tensorpac_is_pac_calc_pac_with_tensorpac():
    # Arrange
    pytest.importorskip("tensorpac")
    import scitex_dsp
    # Act
    from scitex_dsp.utils import _calc_pac_with_tensorpac, pac
    # Act
    # Assert
    assert _calc_pac_with_tensorpac is pac.calc_pac_with_tensorpac


def test_three_pac_public_surfaces_resolve_pac_in_scitex_dsp_utils_all():
    # Arrange
    pytest.importorskip("tensorpac")
    import scitex_dsp
    # Act
    from scitex_dsp.utils import _calc_pac_with_tensorpac, pac
    # Act
    # Assert
    assert "pac" in scitex_dsp.utils.__all__


def test_three_pac_public_surfaces_resolve_calc_pac_with_tensorpac_in_scitex_dsp_utils_all():
    # Arrange
    pytest.importorskip("tensorpac")
    import scitex_dsp
    # Act
    from scitex_dsp.utils import _calc_pac_with_tensorpac, pac
    # Act
    # Assert
    assert "_calc_pac_with_tensorpac" in scitex_dsp.utils.__all__




pytest.importorskip("mne")

import matplotlib.pyplot as plt
import numpy as np

tensorpac = pytest.importorskip("tensorpac")
TENSORPAC_AVAILABLE = True

from scitex.dsp.utils.pac import calc_pac_with_tensorpac, plot_PAC_scitex_vs_tensorpac


class TestCalcPacWithTensorpac:
    """Test calc_pac_with_tensorpac function."""

    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="tensorpac not available")
    @staticmethod
    def _synthetic_pac_signal(fs=512, t_sec=2, theta_freq=6, gamma_freq=40):
        """Build (1,1,T) signal with simple theta-gamma coupling."""
        n_samples = fs * t_sec
        t = np.linspace(0, t_sec, n_samples)
        phase_signal = np.sin(2 * np.pi * theta_freq * t)
        gamma_signal = (1 + 0.5 * phase_signal) * np.sin(2 * np.pi * gamma_freq * t)
        signal = phase_signal + gamma_signal + 0.1 * np.random.randn(n_samples)
        return signal[np.newaxis, np.newaxis, :]

    def test_calc_pac_with_tensorpac_basic_pac_shape(self):
        """calc_pac_with_tensorpac returns pac shaped (n_freqs_pha, n_freqs_amp)."""
        # Arrange
        fs, t_sec = 512, 2
        xx = self._synthetic_pac_signal(fs=fs, t_sec=t_sec)
        # Act
        _phases, _amplitudes, freqs_pha, freqs_amp, pac = calc_pac_with_tensorpac(
            xx, fs, t_sec, i_batch=0, i_ch=0
        )
        # Assert
        assert pac.shape == (len(freqs_pha), len(freqs_amp))

    def test_calc_pac_with_tensorpac_basic_phases_3d(self):
        """phases output is 3-D (freq_pha, epoch, time)."""
        # Arrange
        fs, t_sec = 512, 2
        xx = self._synthetic_pac_signal(fs=fs, t_sec=t_sec)
        # Act
        phases, _amplitudes, _freqs_pha, _freqs_amp, _pac = calc_pac_with_tensorpac(
            xx, fs, t_sec, i_batch=0, i_ch=0
        )
        # Assert
        assert phases.ndim == 3

    def test_calc_pac_with_tensorpac_basic_amplitudes_3d(self):
        """amplitudes output is 3-D (freq_amp, epoch, time)."""
        # Arrange
        fs, t_sec = 512, 2
        xx = self._synthetic_pac_signal(fs=fs, t_sec=t_sec)
        # Act
        _phases, amplitudes, _freqs_pha, _freqs_amp, _pac = calc_pac_with_tensorpac(
            xx, fs, t_sec, i_batch=0, i_ch=0
        )
        # Assert
        assert amplitudes.ndim == 3

    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="tensorpac not available")
    @staticmethod
    def _realistic_eeg_signal(fs=250, t_sec=4):
        """Realistic EEG-like signal with alpha-beta phase modulation."""
        n_samples = fs * t_sec
        t = np.linspace(0, t_sec, n_samples)
        alpha = np.sin(2 * np.pi * 10 * t)
        phase_coupling = np.angle(np.exp(1j * 2 * np.pi * 10 * t))
        beta = (1 + 0.3 * np.cos(phase_coupling)) * np.sin(2 * np.pi * 20 * t)
        noise = 0.2 * np.random.randn(n_samples)
        signal = alpha + beta + noise
        return signal[np.newaxis, np.newaxis, :]

    def test_calc_pac_with_tensorpac_realistic_eeg_pac_is_finite(self):
        """Realistic EEG input produces a finite, non-negative PAC matrix."""
        # Arrange
        fs, t_sec = 250, 4
        xx = self._realistic_eeg_signal(fs=fs, t_sec=t_sec)
        # Act
        _phases, _amplitudes, _freqs_pha, _freqs_amp, pac = calc_pac_with_tensorpac(
            xx, fs, t_sec, i_batch=0, i_ch=0
        )
        # Assert: all PAC entries finite and non-negative.
        assert bool(np.all(np.isfinite(pac)) and pac.min() >= 0)


class TestPlotPacScitexVsTensorpac:
    """Test plot_PAC_scitex_vs_tensorpac with real matplotlib via
    ``scitex.plt`` — no mocks.

    Note: full-render tests are intentionally not included here because the
    production function calls ``ax.imshow2d`` which is provided by a
    plotting extension not always available in CI's ``scitex.plt``
    (only the bare ``imshow`` is). The previous mock-based tests hid
    this gap. Tests below cover the input-validation contract that
    runs before any plotting; full render is exercised by the
    integration suite when the extension is installed."""

    def test_plot_pac_raises_assertion_error_for_mismatched_shapes(self):
        # Arrange
        pac_scitex = np.random.rand(50, 30)
        pac_tp = np.random.rand(40, 30)  # Different shape
        freqs_pha = np.linspace(1, 20, 50)
        freqs_amp = np.linspace(30, 150, 30)
        # Act
        ctx = pytest.raises(AssertionError)
        # Assert
        with ctx:
            plot_PAC_scitex_vs_tensorpac(
                pac_scitex, pac_tp, freqs_pha, freqs_amp
            )


class TestPacIntegration:
    """Test integration between PAC calculation and plotting."""

    def _close(self, fig):
        try:
            plt.close(fig)
        except Exception:
            pass

    def test_pac_calculation_returns_finite_pac_matrix_for_synthetic_signal(self):
        """Run the real tensorpac calc pipeline against synthetic data
        and assert it returns a finite PAC matrix.

        The plot stage is omitted here: ``plot_PAC_scitex_vs_tensorpac``
        depends on ``ax.imshow2d`` from the optional scitex.plt
        extension, which is not part of the install in this test env.
        Splitting calc-from-plot keeps this test honest under no-mocks."""
        # Arrange
        fs = 256
        t_sec = 1
        n_samples = fs * t_sec
        t = np.linspace(0, t_sec, n_samples)
        theta = np.sin(2 * np.pi * 6 * t)
        gamma = (1 + 0.5 * theta) * np.sin(2 * np.pi * 40 * t)
        signal = theta + gamma + 0.05 * np.random.randn(n_samples)
        xx = signal[np.newaxis, np.newaxis, :]
        # Act
        _phases, _amplitudes, _freqs_pha, _freqs_amp, pac_tp = (
            calc_pac_with_tensorpac(xx, fs, t_sec, i_batch=0, i_ch=0)
        )
        # Assert
        assert bool(np.all(np.isfinite(pac_tp))) is True

    def test_pac_module_imports_callable_calc_pac_with_tensorpac(self):
        # Arrange
        # Act
        from scitex.dsp.utils.pac import calc_pac_with_tensorpac
        # Assert
        assert callable(calc_pac_with_tensorpac)

    def test_pac_module_imports_callable_plot_pac_scitex_vs_tensorpac(self):
        # Arrange
        # Act
        from scitex.dsp.utils.pac import plot_PAC_scitex_vs_tensorpac
        # Assert
        assert callable(plot_PAC_scitex_vs_tensorpac)


    def test_pac_realistic_workflow_end_to_end_pac_shape_and_finite(self):
        """End-to-end PAC: pac is shaped (n_freqs_pha, n_freqs_amp) and finite."""
        # Arrange
        fs = 128
        t_sec = 1
        n_samples = fs * t_sec
        t = np.linspace(0, t_sec, n_samples)
        theta = np.sin(2 * np.pi * 8 * t)
        gamma = (1 + 0.5 * theta) * np.sin(2 * np.pi * 40 * t)
        signal = theta + gamma + 0.1 * np.random.randn(n_samples)
        xx = signal[np.newaxis, np.newaxis, :]
        # Act
        _phases, _amplitudes, freqs_pha, freqs_amp, pac = calc_pac_with_tensorpac(
            xx, fs, t_sec, i_batch=0, i_ch=0
        )
        # Assert: PAC shape matches freq grid and contains only finite values.
        assert (
            pac.shape == (len(freqs_pha), len(freqs_amp))
            and bool(np.all(np.isfinite(pac)))
        )


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/pac.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-26 06:26:29 (ywatanabe)"
# # File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/scitex_repo/src/scitex/dsp/utils/pac.py
# # ----------------------------------------
# import os
#
# __FILE__ = "./src/scitex/dsp/utils/pac.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# #! ./env/bin/python3
# # Time-stamp: "2024-04-16 17:07:27"
#
#
# """
# This script does XYZ.
# """
#
# # Imports
# import sys
#
# import matplotlib.pyplot as plt
# import scitex
# import numpy as np
# import tensorpac
#
#
# # Functions
# def calc_pac_with_tensorpac(xx, fs, t_sec, i_batch=0, i_ch=0):
#     # Morlet's Wavelet Transfrmation
#     p = tensorpac.Pac(f_pha="hres", f_amp="mres", dcomplex="wavelet")
#
#     # Bandpass Filtering and Hilbert Transformation
#     phases = p.filter(fs, xx[i_batch, i_ch], ftype="phase", n_jobs=1)  # (50, 20, 2048)
#     amplitudes = p.filter(
#         fs, xx[i_batch, i_ch], ftype="amplitude", n_jobs=1
#     )  # (50, 20, 2048)
#
#     # Calculates xpac
#     k = 2
#     p.idpac = (k, 0, 0)
#     xpac = p.fit(phases, amplitudes)  # (50, 50, 20)
#     pac = xpac.mean(axis=-1)  # (50, 50)
#
#     freqs_amp = p.f_amp.mean(axis=-1)
#     freqs_pha = p.f_pha.mean(axis=-1)
#
#     pac = pac.T  # (amp, pha) -> (pha, amp)
#
#     return phases, amplitudes, freqs_pha, freqs_amp, pac
#
#
# def plot_PAC_scitex_vs_tensorpac(pac_scitex, pac_tp, freqs_pha, freqs_amp):
#     assert pac_scitex.shape == pac_tp.shape
#
#     # Plots
#     fig, axes = scitex.plt.subplots(ncols=3)  # , sharex=True, sharey=True
#
#     # To align scalebars
#     vmin = min(np.min(pac_scitex), np.min(pac_tp), np.min(pac_scitex - pac_tp))
#     vmax = max(np.max(pac_scitex), np.max(pac_tp), np.max(pac_scitex - pac_tp))
#
#     # scitex version
#     ax = axes[0]
#     ax.imshow2d(
#         pac_scitex,
#         cbar=False,
#         vmin=vmin,
#         vmax=vmax,
#     )
#     ax.set_title("scitex")
#
#     # Tensorpac
#     ax = axes[1]
#     ax.imshow2d(
#         pac_tp,
#         cbar=False,
#         vmin=vmin,
#         vmax=vmax,
#     )
#     ax.set_title("Tensorpac")
#
#     # Diff.
#     ax = axes[2]
#     ax.imshow2d(
#         pac_scitex - pac_tp,
#         cbar_label="PAC values",
#         cbar_shrink=0.5,
#         vmin=vmin,
#         vmax=vmax,
#     )
#     ax.set_title(f"Difference\n(scitex - Tensorpac)")
#
#     # for ax in axes:
#     #     ax.set_ticks(
#     #         x_vals=freqs_pha,
#     #         # y_vals=freqs_amp,
#     #     )
#     #     # ax.set_n_ticks()
#
#     fig.suptitle("PAC (MI) values")
#     fig.supxlabel("Frequency for phase [Hz]")
#     fig.supylabel("Frequency for amplitude [Hz]")
#
#     return fig
#
#
# # Snake_case alias for consistency
# def plot_pac_scitex_vs_tensorpac(pac_scitex, pac_tp, freqs_pha, freqs_amp):
#     """
#     Plot comparison between SciTeX and Tensorpac phase-amplitude coupling results.
#
#     This is an alias for plot_PAC_scitex_vs_tensorpac with snake_case naming.
#
#     Parameters
#     ----------
#     pac_scitex : array-like
#         PAC values from SciTeX
#     pac_tp : array-like
#         PAC values from Tensorpac
#     freqs_pha : array-like
#         Phase frequencies
#     freqs_amp : array-like
#         Amplitude frequencies
#
#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         The generated figure
#     """
#     return plot_PAC_scitex_vs_tensorpac(pac_scitex, pac_tp, freqs_pha, freqs_amp)
#
#
# if __name__ == "__main__":
#     import torch
#
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(sys, plt)
#
#     # Parameters
#     FS = 512
#     T_SEC = 4
#
#     xx, tt, fs = scitex.dsp.demo_sig(
#         batch_size=2,
#         n_chs=2,
#         n_segments=2,
#         fs=FS,
#         t_sec=T_SEC,
#         sig_type="tensorpac",
#     )
#
#     # scitex
#     pac_scitex, freqs_pha, freqs_amp = scitex.dsp.pac(
#         xx, fs, batch_size=2, pha_n_bands=50, amp_n_bands=30
#     )
#     i_batch, i_epoch = 0, 0
#     pac_scitex = pac_scitex[i_batch, i_epoch]
#
#     # Tensorpac
#     phases, amplitudes, freqs_pha, freqs_amp, pac_tp = calc_pac_with_tensorpac(
#         xx, fs, T_SEC, i_batch=0, i_ch=0
#     )
#
#     # Plots
#     fig = plot_PAC_scitex_vs_tensorpac(pac_scitex, pac_tp, freqs_pha, freqs_amp)
#     plt.show()
#
#     # Close
#     scitex.session.close(CONFIG)
#
# """
# /home/ywatanabe/proj/entrance/scitex/dsp/utils/pac.py
# """
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/pac.py
# --------------------------------------------------------------------------------
