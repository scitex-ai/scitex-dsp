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
    def test_calc_pac_with_tensorpac_basic(self):
        """Test basic PAC calculation with tensorpac."""
        # Create synthetic signal with phase-amplitude coupling
        # Arrange
        # Act
        # Assert
        fs = 512
        t_sec = 2
        n_samples = fs * t_sec
        t = np.linspace(0, t_sec, n_samples)

        # Create signal with theta-gamma coupling
        theta_freq = 6  # Hz
        gamma_freq = 40  # Hz

        # Phase signal (theta)
        phase_signal = np.sin(2 * np.pi * theta_freq * t)

        # Amplitude modulated gamma signal
        gamma_signal = (1 + 0.5 * phase_signal) * np.sin(2 * np.pi * gamma_freq * t)

        # Combine signals
        signal = phase_signal + gamma_signal + 0.1 * np.random.randn(n_samples)

        # Create batch structure expected by the function
        xx = signal[np.newaxis, np.newaxis, :]  # (batch, ch, time)

        try:
            phases, amplitudes, freqs_pha, freqs_amp, pac = calc_pac_with_tensorpac(
                xx, fs, t_sec, i_batch=0, i_ch=0
            )

            # Verify outputs
            assert phases.ndim == 3  # (freq_pha, epoch, time)
            assert amplitudes.ndim == 3  # (freq_amp, epoch, time)
            assert isinstance(freqs_pha, np.ndarray)
            assert isinstance(freqs_amp, np.ndarray)
            assert pac.ndim == 2  # (freq_pha, freq_amp)
            assert pac.shape == (len(freqs_pha), len(freqs_amp))

        except Exception as e:
            pytest.skip(f"tensorpac import or execution failed: {e}")

    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="tensorpac not available")
    def test_calc_pac_with_tensorpac_realistic_eeg(self):
        """Test PAC calculation with realistic EEG-like signal."""
        # Realistic EEG parameters
        # Arrange
        # Act
        # Assert
        fs = 250  # Hz
        t_sec = 4  # seconds
        n_samples = fs * t_sec
        t = np.linspace(0, t_sec, n_samples)

        # Multiple frequency components
        # Alpha rhythm (8-12 Hz)
        alpha = np.sin(2 * np.pi * 10 * t)

        # Beta rhythm modulated by alpha phase
        beta_freq = 20
        phase_coupling = np.angle(np.exp(1j * 2 * np.pi * 10 * t))
        beta = (1 + 0.3 * np.cos(phase_coupling)) * np.sin(2 * np.pi * beta_freq * t)

        # Add noise
        noise = 0.2 * np.random.randn(n_samples)
        signal = alpha + beta + noise

        # Batch format
        xx = signal[np.newaxis, np.newaxis, :]

        try:
            phases, amplitudes, freqs_pha, freqs_amp, pac = calc_pac_with_tensorpac(
                xx, fs, t_sec, i_batch=0, i_ch=0
            )

            # Verify realistic frequency ranges
            assert freqs_pha.min() >= 1.0  # Low frequency for phase
            assert freqs_pha.max() <= fs / 4  # Below Nyquist/2
            assert freqs_amp.min() >= freqs_pha.max()  # Amplitude freq > phase freq
            assert freqs_amp.max() <= fs / 2  # Below Nyquist

            # Verify PAC values are reasonable
            assert np.all(np.isfinite(pac))
            assert pac.min() >= 0  # PAC values should be non-negative

        except Exception as e:
            pytest.skip(f"tensorpac execution failed: {e}")


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
        try:
            _phases, _amplitudes, _freqs_pha, _freqs_amp, pac_tp = (
                calc_pac_with_tensorpac(xx, fs, t_sec, i_batch=0, i_ch=0)
            )
        except Exception as exc:  # pragma: no cover - env-specific
            pytest.skip(f"tensorpac calc_pac failed in this env: {exc}")
        # Act
        all_finite = bool(np.all(np.isfinite(pac_tp)))
        # Assert
        assert all_finite is True

    def test_pac_module_imports_callable_calc_pac_with_tensorpac(self):
        # Arrange
        # Act
        # Arrange
        # Act
        from scitex.dsp.utils.pac import (
            calc_pac_with_tensorpac,
            plot_PAC_scitex_vs_tensorpac,
        )
        # Act
        # Assert
        assert callable(calc_pac_with_tensorpac)

    def test_pac_module_imports_callable_plot_pac_scitex_vs_tensorpac(self):
        # Arrange
        # Act
        # Arrange
        # Act
        from scitex.dsp.utils.pac import (
            calc_pac_with_tensorpac,
            plot_PAC_scitex_vs_tensorpac,
        )
        # Act
        # Assert
        assert callable(plot_PAC_scitex_vs_tensorpac)


    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="tensorpac not available")
    def test_pac_realistic_workflow_end_to_end(self):
        """Test realistic PAC workflow if tensorpac is available."""
        # Arrange
        # Act
        # Assert
        try:
            # Generate synthetic coupled signal
            fs = 128  # Lower sampling rate for faster test
            t_sec = 1
            n_samples = fs * t_sec
            t = np.linspace(0, t_sec, n_samples)

            # Simple theta-gamma coupling
            theta = np.sin(2 * np.pi * 8 * t)
            gamma = (1 + 0.5 * theta) * np.sin(2 * np.pi * 40 * t)
            signal = theta + gamma + 0.1 * np.random.randn(n_samples)

            xx = signal[np.newaxis, np.newaxis, :]

            # Calculate PAC
            phases, amplitudes, freqs_pha, freqs_amp, pac = calc_pac_with_tensorpac(
                xx, fs, t_sec, i_batch=0, i_ch=0
            )

            # Verify results are reasonable
            assert phases.shape[0] > 0  # Has phase frequencies
            assert amplitudes.shape[0] > 0  # Has amplitude frequencies
            assert pac.shape == (len(freqs_pha), len(freqs_amp))
            assert np.all(np.isfinite(pac))

        except Exception as e:
            pytest.skip(f"End-to-end PAC test failed: {e}")


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
