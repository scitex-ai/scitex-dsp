#!/usr/bin/env python3
# Time-stamp: "2025-06-02 15:45:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/utils/test__differential_bandpass_filters.py

"""Tests for differential bandpass filters."""

import os

import pytest

torch = pytest.importorskip("torch")
# Module-level guard: most tests below exercise build_bandpass_filters which
# pulls in torchaudio.prototype.functional.sinc_impulse_response. We use
# importorskip up front so individual tests don't each need their own
# try/except/skip (which would trip TQ007 on every test).
pytest.importorskip("torchaudio.prototype.functional")

import numpy as np
import torch.nn as nn


class TestDifferentialBandpassFiltersAvailableFlags:
    """Test _AVAILABLE flags for optional dependencies."""

    def test_torch_available_flag_exists(self):
        """Test that TORCH_AVAILABLE flag is exported."""
        # Arrange
        # Act
        from scitex.dsp.utils._differential_bandpass_filters import TORCH_AVAILABLE

        # Assert
        assert isinstance(TORCH_AVAILABLE, bool)

    def test_torchaudio_available_flag_exists(self):
        """Test that TORCHAUDIO_AVAILABLE flag is exported."""
        # Arrange
        # Act
        from scitex.dsp.utils._differential_bandpass_filters import TORCHAUDIO_AVAILABLE

        # Assert
        assert isinstance(TORCHAUDIO_AVAILABLE, bool)

    def test_check_torch_function_exists(self):
        """Test that _check_torch function is exported."""
        # Arrange
        # Act
        from scitex.dsp.utils._differential_bandpass_filters import _check_torch

        # Assert
        assert callable(_check_torch)

    def test_check_sinc_available_function_exists(self):
        """Test that _check_sinc_available function is exported."""
        # Arrange
        # Act
        from scitex.dsp.utils._differential_bandpass_filters import (
            _check_sinc_available,
        )

        # Assert
        assert callable(_check_sinc_available)

    def test_torch_available_is_true_when_torch_installed(self):
        """Test that TORCH_AVAILABLE is True when torch is installed."""
        # Arrange
        # Act
        from scitex.dsp.utils._differential_bandpass_filters import TORCH_AVAILABLE

        # Assert
        assert TORCH_AVAILABLE is True

    def test_check_torch_does_not_raise_when_available(self):
        """Test that _check_torch doesn't raise when torch is available."""
        # Arrange
        from scitex.dsp.utils._differential_bandpass_filters import _check_torch
        # Act
        result = _check_torch()
        # Assert
        assert result is None


def test_init_bandpass_filters_basic_returns_tensor_and_parameters():
    """init_bandpass_filters returns filters tensor and learnable mids in
    the default frequency bands (2-20 Hz phase, 60-160 Hz amplitude)."""
    # Arrange
    from scitex.dsp.utils import init_bandpass_filters
    sig_len, fs = 1000, 250
    # Act
    filters, pha_mids, amp_mids = init_bandpass_filters(sig_len, fs)
    # Assert: tensor/parameter types + default band ranges all hold.
    assert (
        isinstance(filters, torch.Tensor)
        and isinstance(pha_mids, nn.Parameter)
        and isinstance(amp_mids, nn.Parameter)
        and pha_mids.min() >= 2
        and pha_mids.max() <= 20
        and amp_mids.min() >= 60
        and amp_mids.max() <= 160
    )


def test_init_bandpass_filters_custom_params_shapes_and_ranges():
    """init_bandpass_filters honours custom n_bands and frequency ranges."""
    # Arrange
    from scitex.dsp.utils import init_bandpass_filters
    sig_len, fs = 2000, 500
    pha_low_hz, pha_high_hz, pha_n_bands = 1, 30, 20
    amp_low_hz, amp_high_hz, amp_n_bands = 50, 200, 40
    cycle = 5
    # Act
    filters, pha_mids, amp_mids = init_bandpass_filters(
        sig_len, fs, pha_low_hz, pha_high_hz, pha_n_bands,
        amp_low_hz, amp_high_hz, amp_n_bands, cycle,
    )
    # Assert: per-band counts, frequency ranges, and total filter row count.
    assert (
        pha_mids.shape == (pha_n_bands,)
        and amp_mids.shape == (amp_n_bands,)
        and pha_mids.min() >= pha_low_hz
        and pha_mids.max() <= pha_high_hz
        and amp_mids.min() >= amp_low_hz
        and amp_mids.max() <= amp_high_hz
        and filters.shape[0] == pha_n_bands + amp_n_bands
    )


def test_build_bandpass_filters_basic_shape_is_2d_and_odd_length():
    """build_bandpass_filters returns a 2-D tensor with one odd-length
    impulse response per phase + amplitude band."""
    # Arrange
    from scitex.dsp.utils import build_bandpass_filters
    sig_len, fs = 1000, 250
    pha_mids = torch.linspace(2, 20, 10)
    amp_mids = torch.linspace(60, 160, 15)
    cycle = 3
    # Act
    filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle)
    # Assert
    assert (
        isinstance(filters, torch.Tensor)
        and filters.ndim == 2
        and filters.shape[0] == len(pha_mids) + len(amp_mids)
        and filters.shape[1] % 2 == 1
    )


def test_build_bandpass_filters_gradients_flow_to_both_mids():
    """Gradients flow through build_bandpass_filters to both pha_mids and
    amp_mids and are non-zero (verifies the filter band-edges remain
    differentiable end-to-end)."""
    # Arrange
    from scitex.dsp.utils import build_bandpass_filters
    sig_len, fs, cycle = 500, 250, 3
    pha_mids = torch.linspace(2, 20, 5, requires_grad=True)
    amp_mids = torch.linspace(60, 160, 8, requires_grad=True)
    # Act
    filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle)
    filters.sum().backward()
    # Assert: both grads exist and are nonzero.
    assert (
        pha_mids.grad is not None
        and amp_mids.grad is not None
        and not torch.allclose(pha_mids.grad, torch.zeros_like(pha_mids.grad))
        and not torch.allclose(amp_mids.grad, torch.zeros_like(amp_mids.grad))
    )


def test_build_bandpass_filters_torch_fn_decorator_returns_tensor_for_numpy_input():
    """torch_fn decorator converts numpy mids inputs and still returns torch tensor."""
    # Arrange
    from scitex.dsp.utils import build_bandpass_filters
    sig_len, fs, cycle = 500, 250, 3
    pha_mids = np.linspace(2, 20, 5)
    amp_mids = np.linspace(60, 160, 8)
    # Act
    filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle)
    # Assert
    assert isinstance(filters, torch.Tensor)


def test_bandpass_filters_real_eeg_scenario_bands_in_alpha_beta_and_gamma():
    """Realistic EEG: filters cover alpha-beta phase bands and gamma
    amplitude bands, learnable, and total filter count matches request."""
    # Arrange
    from scitex.dsp.utils import init_bandpass_filters
    fs = 250
    sig_len = int(fs * 2.0)
    pha_low_hz, pha_high_hz, pha_n_bands = 8, 30, 20
    amp_low_hz, amp_high_hz, amp_n_bands = 30, 100, 30
    cycle = 3
    # Act
    filters, pha_mids, amp_mids = init_bandpass_filters(
        sig_len, fs, pha_low_hz, pha_high_hz, pha_n_bands,
        amp_low_hz, amp_high_hz, amp_n_bands, cycle,
    )
    # Assert: total count, band ranges, and learnability all hold.
    assert (
        filters.shape[0] == pha_n_bands + amp_n_bands
        and pha_mids.min() >= 8.0
        and pha_mids.max() <= 30.0
        and amp_mids.min() >= 30.0
        and amp_mids.max() <= 100.0
        and pha_mids.requires_grad
        and amp_mids.requires_grad
    )


def test_bandpass_filters_different_signal_lengths_filter_length_bounded():
    """build_bandpass_filters produces tensor-shaped output with
    filters.shape[0] = n_pha + n_amp and filters.shape[1] <= sig_len for
    a range of signal lengths."""
    # Arrange
    from scitex.dsp.utils import build_bandpass_filters
    fs, cycle = 250, 3
    pha_mids = torch.linspace(2, 20, 5)
    amp_mids = torch.linspace(60, 160, 5)
    signal_lengths = [100, 500, 1000, 2000]
    # Act
    outputs = [
        (sig_len, build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle))
        for sig_len in signal_lengths
    ]
    n_bands_expected = len(pha_mids) + len(amp_mids)
    # Assert: type / band-count / length-bound all hold for every sig_len.
    assert all(
        isinstance(f, torch.Tensor)
        and f.shape[0] == n_bands_expected
        and f.shape[1] <= sig_len
        for sig_len, f in outputs
    )


def test_bandpass_filters_edge_cases_single_band_returns_two_filters():
    """Single phase + single amplitude band -> 2 filter rows."""
    # Arrange
    from scitex.dsp.utils import build_bandpass_filters
    sig_len, fs, cycle = 1000, 250, 3
    pha_mids = torch.tensor([10.0])
    amp_mids = torch.tensor([80.0])
    # Act
    filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle)
    # Assert
    assert filters.shape[0] == 2


def test_bandpass_filters_edge_cases_very_short_signal_filter_length_bounded():
    """build_bandpass_filters fits filter length within a very short signal."""
    # Arrange
    from scitex.dsp.utils import build_bandpass_filters
    sig_len, fs, cycle = 100, 250, 3
    pha_mids = torch.tensor([10.0])
    amp_mids = torch.tensor([80.0])
    # Act
    filters_short = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle)
    # Assert
    assert filters_short.shape[1] <= sig_len


def test_bandpass_filters_parameter_validation_overlapping_pha_amp_ranges():
    """build_bandpass_filters accepts overlapping pha/amp frequency ranges."""
    # Arrange
    from scitex.dsp.utils import build_bandpass_filters
    sig_len, fs, cycle = 1000, 250, 3
    pha_mids = torch.linspace(5, 25, 5)
    amp_mids = torch.linspace(20, 100, 5)  # Overlap with pha range
    # Act
    filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle)
    # Assert
    assert isinstance(filters, torch.Tensor)


def test_bandpass_filters_parameter_validation_wide_frequency_range():
    """build_bandpass_filters accepts a wide phase frequency range (1-50 Hz)."""
    # Arrange
    from scitex.dsp.utils import build_bandpass_filters
    sig_len, fs, cycle = 1000, 250, 3
    pha_mids = torch.linspace(1, 50, 10)
    amp_mids = torch.linspace(60, 120, 10)
    # Act
    filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle)
    # Assert
    assert isinstance(filters, torch.Tensor)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/_differential_bandpass_filters.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-11-26 22:24:13 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/utils/_differential_bandpass_filters.py
#
# THIS_FILE = "/home/ywatanabe/proj/scitex_repo/src/scitex/dsp/utils/_differential_bandpass_filters.py"
#
# import sys
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# try:
#     import torch
#     import torch.nn as nn
#
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#     torch = None
#     nn = None
#
# from scitex.decorators import torch_fn
# from scitex.gen._to_even import to_even
# from scitex.gen._to_odd import to_odd
#
# try:
#     from torchaudio.prototype.functional import sinc_impulse_response
#
#     TORCHAUDIO_AVAILABLE = True
# except ImportError:
#     TORCHAUDIO_AVAILABLE = False
#     sinc_impulse_response = None
#
#
# def _check_torch():
#     if not TORCH_AVAILABLE:
#         raise ImportError(
#             "PyTorch is not installed. Please install with: pip install torch"
#         )
#
#
# def _check_sinc_available():
#     if sinc_impulse_response is None:
#         raise ImportError(
#             "sinc_impulse_response requires torchaudio.prototype.functional. "
#             "Install torchaudio with: pip install torchaudio"
#         )
#
#
# # Functions
# @torch_fn
# def init_bandpass_filters(
#     sig_len,
#     fs,
#     pha_low_hz=2,
#     pha_high_hz=20,
#     pha_n_bands=30,
#     amp_low_hz=60,
#     amp_high_hz=160,
#     amp_n_bands=50,
#     cycle=3,
# ):
#     _check_sinc_available()
#     # Learnable parameters
#     pha_mids = nn.Parameter(torch.linspace(pha_low_hz, pha_high_hz, pha_n_bands))
#     amp_mids = nn.Parameter(torch.linspace(amp_low_hz, amp_high_hz, amp_n_bands))
#     filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle)
#     return filters, pha_mids, amp_mids
#
#
# @torch_fn
# def build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle):
#     _check_sinc_available()
#
#     def _define_freqs(mids, factor):
#         lows = mids - mids / factor
#         highs = mids + mids / factor
#         return lows, highs
#
#     def define_order(low_hz, fs, sig_len, cycle):
#         order = cycle * int(fs // low_hz)
#         order = order if 3 * order >= sig_len else (sig_len - 1) // 3
#         order = to_even(order)
#         return order
#
#     def _calc_filters(lows_hz, highs_hz, fs, order):
#         nyq = fs / 2.0
#         order = to_odd(order)
#         # lowpass filters
#         irs_ll = sinc_impulse_response(lows_hz / nyq, window_size=order)
#         irs_hh = sinc_impulse_response(highs_hz / nyq, window_size=order)
#         irs = irs_ll - irs_hh
#         return irs
#
#     # Main
#     pha_lows, pha_highs = _define_freqs(pha_mids, factor=4.0)
#     amp_lows, amp_highs = _define_freqs(amp_mids, factor=8.0)
#
#     lowest = min(pha_lows.min().item(), amp_lows.min().item())
#     order = define_order(lowest, fs, sig_len, cycle)
#
#     pha_bp_filters = _calc_filters(pha_lows, pha_highs, fs, order)
#     amp_bp_filters = _calc_filters(amp_lows, amp_highs, fs, order)
#     return torch.vstack([pha_bp_filters, amp_bp_filters])
#
#
# if __name__ == "__main__":
#     import scitex
#
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(sys, plt, agg=True)
#
#     # Demo signal
#     freqs_hz = [10, 30, 100, 300]
#     fs = 1024
#     xx, tt, fs = scitex.dsp.demo_sig(fs=fs, freqs_hz=freqs_hz)
#
#     # Main
#     filters, pha_mids, amp_mids = init_bandpass_filters(xx.shape[-1], fs)
#
#     filters.sum().backward()  # OK. The filtering bands are trainable with backpropagation.
#
#     # Update 'pha_mids' and 'amp_mids' in the forward method.
#     # Then, re-build filters using optimized parameters like this:
#     # self.filters = build_bandpass_filters(self.sig_len, self.fs, self.pha_mids, self.amp_mids, self.cycle)
#
#     mids_all = np.concatenate(
#         [pha_mids.detach().cpu().numpy(), amp_mids.detach().cpu().numpy()]
#     )
#
#     for i_filter in range(len(mids_all)):
#         mid = mids_all[i_filter]
#         fig = scitex.dsp.utils.filter.plot_filter_responses(
#             filters[i_filter].detach().cpu().numpy(), fs, title=f"{mid:.1f} Hz"
#         )
#         scitex.io.save(
#             fig,
#             f"differentiable_bandpass_filter_reponses_filter#{i_filter:03d}_{mid:.1f}_Hz.png",
#         )
#     # plt.show()
#
# # EOF
#
# """
# python -m scitex.dsp.utils._differential_bandpass_filters
# """
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/_differential_bandpass_filters.py
# --------------------------------------------------------------------------------
