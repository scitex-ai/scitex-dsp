#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:24:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dsp/utils/test_filter.py

"""Tests for filter functionality."""

import pytest

pytest.importorskip("mne")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

import scitex
from scitex.dsp.utils.filter import design_filter, plot_filter_responses


class TestDesignFilter:
    """Test design_filter function."""

    def test_design_filter_lowpass_filter_coeffs_is_np_ndarray(self):
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 30.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, low_hz=low_hz)
        # Assert
        assert isinstance(filter_coeffs, np.ndarray)

    def test_design_filter_lowpass_len_filter_coeffs_0(self):
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 30.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, low_hz=low_hz)
        # Assert
        assert len(filter_coeffs) > 0

    def test_design_filter_lowpass_filter_coeffs_dtype_in_np_float32_np_float64(self):
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 30.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, low_hz=low_hz)
        # Assert
        assert filter_coeffs.dtype in [np.float32, np.float64]


    def test_design_filter_highpass_filter_coeffs_is_np_ndarray(self):
        # Arrange
        sig_len = 1000
        fs = 250
        high_hz = 70.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, high_hz=high_hz)
        # Assert
        assert isinstance(filter_coeffs, np.ndarray)

    def test_design_filter_highpass_len_filter_coeffs_0(self):
        # Arrange
        sig_len = 1000
        fs = 250
        high_hz = 70.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, high_hz=high_hz)
        # Assert
        assert len(filter_coeffs) > 0

    def test_design_filter_highpass_filter_coeffs_dtype_in_np_float32_np_float64(self):
        # Arrange
        sig_len = 1000
        fs = 250
        high_hz = 70.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, high_hz=high_hz)
        # Assert
        assert filter_coeffs.dtype in [np.float32, np.float64]


    def test_design_filter_bandpass_filter_coeffs_is_np_ndarray(self):
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 8.0
        high_hz = 30.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, low_hz=low_hz, high_hz=high_hz)
        # Assert
        assert isinstance(filter_coeffs, np.ndarray)

    def test_design_filter_bandpass_len_filter_coeffs_0(self):
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 8.0
        high_hz = 30.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, low_hz=low_hz, high_hz=high_hz)
        # Assert
        assert len(filter_coeffs) > 0

    def test_design_filter_bandpass_filter_coeffs_dtype_in_np_float32_np_float64(self):
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 8.0
        high_hz = 30.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, low_hz=low_hz, high_hz=high_hz)
        # Assert
        assert filter_coeffs.dtype in [np.float32, np.float64]


    def test_design_filter_bandstop_filter_coeffs_is_np_ndarray(self):
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 48.0
        high_hz = 52.0
        # Act
        filter_coeffs = design_filter(
            sig_len, fs, low_hz=low_hz, high_hz=high_hz, is_bandstop=True
        )
        # Assert
        assert isinstance(filter_coeffs, np.ndarray)

    def test_design_filter_bandstop_len_filter_coeffs_0(self):
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 48.0
        high_hz = 52.0
        # Act
        filter_coeffs = design_filter(
            sig_len, fs, low_hz=low_hz, high_hz=high_hz, is_bandstop=True
        )
        # Assert
        assert len(filter_coeffs) > 0

    def test_design_filter_bandstop_filter_coeffs_dtype_in_np_float32_np_float64(self):
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 48.0
        high_hz = 52.0
        # Act
        filter_coeffs = design_filter(
            sig_len, fs, low_hz=low_hz, high_hz=high_hz, is_bandstop=True
        )
        # Assert
        assert filter_coeffs.dtype in [np.float32, np.float64]


    @pytest.mark.parametrize(
        "kwargs",
        [
            {"low_hz": 8.0, "high_hz": 12.0},
            {"low_hz": 13.0, "high_hz": 30.0},
            {"low_hz": 48.0, "high_hz": 52.0, "is_bandstop": True},
        ],
    )
    def test_design_filter_real_eeg_scenario_is_ndarray(self, kwargs):
        """Test filter design with realistic EEG parameters returns ndarray."""
        # Arrange
        fs = 250
        sig_len = fs * 4
        # Act
        filt = design_filter(sig_len, fs, **kwargs)
        # Assert
        assert isinstance(filt, np.ndarray)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"low_hz": 8.0, "high_hz": 12.0},
            {"low_hz": 13.0, "high_hz": 30.0},
            {"low_hz": 48.0, "high_hz": 52.0, "is_bandstop": True},
        ],
    )
    def test_design_filter_real_eeg_scenario_nonempty(self, kwargs):
        """Test filter design with realistic EEG parameters returns nonempty."""
        # Arrange
        fs = 250
        sig_len = fs * 4
        # Act
        filt = design_filter(sig_len, fs, **kwargs)
        # Assert
        assert len(filt) > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"low_hz": 8.0, "high_hz": 12.0},
            {"low_hz": 13.0, "high_hz": 30.0},
            {"low_hz": 48.0, "high_hz": 52.0, "is_bandstop": True},
        ],
    )
    def test_design_filter_real_eeg_scenario_float_dtype(self, kwargs):
        """Test filter design with realistic EEG parameters returns float dtype."""
        # Arrange
        fs = 250
        sig_len = fs * 4
        # Act
        filt = design_filter(sig_len, fs, **kwargs)
        # Assert
        assert filt.dtype in [np.float32, np.float64]

    @pytest.mark.parametrize("cycle", [1, 3, 5])
    def test_design_filter_cycle_parameter_is_ndarray(self, cycle):
        """Test different cycle parameter values return ndarray."""
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 10.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, low_hz=low_hz, cycle=cycle)
        # Assert
        assert isinstance(filter_coeffs, np.ndarray)

    @pytest.mark.parametrize("cycle", [1, 3, 5])
    def test_design_filter_cycle_parameter_nonempty(self, cycle):
        """Test different cycle parameter values return nonempty filter."""
        # Arrange
        sig_len = 1000
        fs = 250
        low_hz = 10.0
        # Act
        filter_coeffs = design_filter(sig_len, fs, low_hz=low_hz, cycle=cycle)
        # Assert
        assert len(filter_coeffs) > 0

    def test_design_filter_frequency_response(self):
        """Test filter frequency response characteristics."""
        # Arrange
        sig_len = 2000
        fs = 500
        low_hz = 10.0
        high_hz = 50.0
        bp_filter = design_filter(sig_len, fs, low_hz=low_hz, high_hz=high_hz)
        w, h = freqz(bp_filter, worN=1024, fs=fs)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        passband_idx = (w >= low_hz) & (w <= high_hz)
        stopband_low_idx = w < low_hz * 0.5
        # Act
        passband_mean = np.mean(magnitude_db[passband_idx])
        stopband_mean = np.mean(magnitude_db[stopband_low_idx])
        # Assert
        assert passband_mean > stopband_mean

    def test_design_filter_edge_cases_short_filter_is_np_ndarray(self):
        # Arrange
        # Act
        short_filter = design_filter(sig_len=100, fs=250, low_hz=10.0)
        # Assert
        assert isinstance(short_filter, np.ndarray)

    def test_design_filter_edge_cases_short_filter_length(self):
        # Arrange
        # Act
        short_filter = design_filter(sig_len=100, fs=250, low_hz=10.0)
        # Assert
        assert len(short_filter) <= 100

    def test_design_filter_edge_cases_high_fs_filter_is_np_ndarray(self):
        # Arrange
        # Act
        high_fs_filter = design_filter(sig_len=1000, fs=2000, low_hz=100.0)
        # Assert
        assert isinstance(high_fs_filter, np.ndarray)


    def test_design_filter_edge_cases_low_freq_filter_is_np_ndarray(self):
        # Arrange
        # Act
        low_freq_filter = design_filter(sig_len=2000, fs=250, low_hz=1.0)
        # Assert
        assert isinstance(low_freq_filter, np.ndarray)



    def test_design_filter_parameter_validation_raises_exception(self):
        # Arrange
        sig_len = 1000
        fs = 250
        # Act
        # Assert
        with pytest.raises(Exception):  # Should raise FilterParameterError
            design_filter(sig_len, fs)

    def test_design_filter_parameter_validation_raises_exception_2(self):
        # Arrange
        sig_len = 1000
        fs = 250
        # Act
        # Assert
        with pytest.raises(Exception):
            design_filter(sig_len, fs, low_hz=-10.0)

    def test_design_filter_parameter_validation_raises_exception_3(self):
        # Arrange
        sig_len = 1000
        fs = 250
        # Act
        # Assert
        with pytest.raises(Exception):
            design_filter(sig_len, fs, high_hz=-10.0)

    def test_design_filter_parameter_validation_raises_exception_4(self):
        # Arrange
        sig_len = 1000
        fs = 250
        # Act
        # Assert
        with pytest.raises(Exception):
            design_filter(sig_len, fs, low_hz=50.0, high_hz=10.0)


    @pytest.mark.parametrize(
        "args,kwargs",
        [
            ((1000, 250), {"low_hz": 10.0}),
            ((1000.0, 250.0), {"low_hz": 10.0}),
            (
                (np.array([1000]), np.array([250])),
                {"low_hz": np.array([10.0])},
            ),
        ],
    )
    def test_design_filter_numpy_conversion_is_ndarray(self, args, kwargs):
        """Test numpy_fn decorator returns ndarray for various input types."""
        # Arrange
        # Act
        filt = design_filter(*args, **kwargs)
        # Assert
        assert isinstance(filt, np.ndarray)

    @pytest.mark.parametrize(
        "args,kwargs",
        [
            ((1000, 250), {"low_hz": 10.0}),
            ((1000.0, 250.0), {"low_hz": 10.0}),
            (
                (np.array([1000]), np.array([250])),
                {"low_hz": np.array([10.0])},
            ),
        ],
    )
    def test_design_filter_numpy_conversion_nonempty(self, args, kwargs):
        """Test numpy_fn decorator returns nonempty for various input types."""
        # Arrange
        # Act
        filt = design_filter(*args, **kwargs)
        # Assert
        assert len(filt) > 0


class TestPlotFilterResponses:
    """Test plot_filter_responses function.

    Uses the real ``scitex.plt`` wrapper around matplotlib rather than
    patching it. The function returns the matplotlib Figure it built;
    asserting that's a real Figure is the contract that matters.
    """

    def _close_after(self, fig):
        try:
            plt.close(fig)
        except Exception:
            pass

    def test_plot_filter_responses_returns_matplotlib_figure_for_simple_filter(self):
        # Arrange
        filter_coeffs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        fs = 250
        # Act
        result = plot_filter_responses(filter_coeffs, fs)
        self._close_after(result)
        # Assert
        assert hasattr(result, "savefig")

    def test_plot_filter_responses_accepts_title_without_raising(self):
        # Arrange
        filter_coeffs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        fs = 250
        # Act
        fig = plot_filter_responses(filter_coeffs, fs, title="Test Filter")
        self._close_after(fig)
        # Assert
        assert hasattr(fig, "savefig")

    def test_plot_filter_responses_accepts_custom_worN_resolution(self):
        # Arrange
        filter_coeffs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        fs = 250
        # Act
        fig = plot_filter_responses(filter_coeffs, fs, worN=4000)
        self._close_after(fig)
        # Assert
        assert hasattr(fig, "savefig")

    def test_plot_filter_responses_works_for_designed_bandpass_filter(self):
        # Arrange
        sig_len = 1000
        fs = 250
        filter_coeffs = design_filter(sig_len, fs, low_hz=10.0, high_hz=40.0)
        # Act
        fig = plot_filter_responses(filter_coeffs, fs)
        self._close_after(fig)
        # Assert
        assert hasattr(fig, "savefig")

    def test_plot_filter_responses_accepts_python_list_input(self):
        # Arrange
        filter_coeffs = [0.1, 0.2, 0.4, 0.2, 0.1]
        fs = 250.0
        # Act
        fig = plot_filter_responses(filter_coeffs, fs)
        self._close_after(fig)
        # Assert
        assert hasattr(fig, "savefig")


class TestFilterIntegration:
    """Integration between filter design and plotting — uses real
    matplotlib via ``scitex.plt``; no mocks."""

    def _close_after(self, fig):
        try:
            plt.close(fig)
        except Exception:
            pass

    def test_design_and_plot_integration_returns_real_figure(self):
        # Arrange
        sig_len = 1000
        fs = 250
        filter_coeffs = design_filter(sig_len, fs, low_hz=8.0, high_hz=30.0)
        # Act
        fig = plot_filter_responses(
            filter_coeffs, fs, title="Alpha-Beta Band Filter"
        )
        self._close_after(fig)
        # Assert
        assert hasattr(fig, "savefig")

    @pytest.mark.parametrize(
        "filter_type,kwargs",
        [
            ("lowpass", {"low_hz": 50.0}),
            ("highpass", {"high_hz": 1.0}),
            ("bandpass", {"low_hz": 8.0, "high_hz": 30.0}),
            ("bandstop", {"low_hz": 48.0, "high_hz": 52.0, "is_bandstop": True}),
        ],
    )
    def test_designed_filter_plot_returns_figure(self, filter_type, kwargs):
        # Arrange
        sig_len = 2000
        fs = 500
        filter_coeffs = design_filter(sig_len, fs, **kwargs)
        # Act
        fig = plot_filter_responses(
            filter_coeffs, fs, title=f"{filter_type.title()} Filter"
        )
        self._close_after(fig)
        # Assert
        assert hasattr(fig, "savefig")


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Source code section truncated for brevity.
# --------------------------------------------------------------------------------
