#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the CPU-safe ``device="auto"`` default in ``pac`` and ``wavelet``.

These verify that both functions run on a CPU-only environment (torch present,
no CUDA) via the new ``device="auto"`` default, that explicit ``device="cpu"``
keeps working, and — when CUDA is present — that ``"auto"`` resolves to cuda.
The auto/cpu paths deliberately do NOT skip on CPU-only runners: that is the
whole point of this change (unblocking neurovista's CPU feature-extraction).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scitex_dsp import pac, wavelet


def _wavelet_sig():
    fs = 256
    t = np.linspace(0, 2, 2 * fs)
    x = np.sin(2 * np.pi * 10 * t).reshape(1, 1, -1).astype(np.float32)
    return x, fs


def _pac_sig():
    fs = 512
    n_samples = int(fs * 2)
    x = np.random.randn(1, 2, n_samples).astype(np.float32)
    return x, fs


class TestWaveletDeviceAuto:
    def test_wavelet_auto_pha_is_ndarray(self):
        # Arrange
        x, fs = _wavelet_sig()
        # Act
        pha, amp, freqs = wavelet(x, fs, device="auto")
        # Assert
        assert isinstance(pha, np.ndarray)

    def test_wavelet_auto_amp_is_ndarray(self):
        # Arrange
        x, fs = _wavelet_sig()
        # Act
        pha, amp, freqs = wavelet(x, fs, device="auto")
        # Assert
        assert isinstance(amp, np.ndarray)

    def test_wavelet_auto_freqs_is_ndarray(self):
        # Arrange
        x, fs = _wavelet_sig()
        # Act
        pha, amp, freqs = wavelet(x, fs, device="auto")
        # Assert
        assert isinstance(freqs, np.ndarray)

    def test_wavelet_auto_batch_dim_preserved(self):
        # Arrange
        x, fs = _wavelet_sig()
        # Act
        pha, amp, freqs = wavelet(x, fs, device="auto")
        # Assert
        assert pha.shape[0] == 1  # batch size

    def test_wavelet_auto_matches_cpu_pha_shape(self):
        # Arrange
        x, fs = _wavelet_sig()
        # Act
        pha_auto, _, _ = wavelet(x, fs)  # default device="auto"
        pha_cpu, _, _ = wavelet(x, fs, device="cpu")
        # Assert
        assert pha_auto.shape == pha_cpu.shape

    def test_wavelet_auto_matches_cpu_amp_shape(self):
        # Arrange
        x, fs = _wavelet_sig()
        # Act
        _, amp_auto, _ = wavelet(x, fs)  # default device="auto"
        _, amp_cpu, _ = wavelet(x, fs, device="cpu")
        # Assert
        assert amp_auto.shape == amp_cpu.shape

    def test_wavelet_explicit_cpu_amp_is_ndarray(self):
        # Arrange
        x, fs = _wavelet_sig()
        # Act
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        # Assert
        assert isinstance(amp, np.ndarray)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_wavelet_auto_picks_cuda_when_available(self):
        # Arrange
        x, fs = _wavelet_sig()
        xt = torch.from_numpy(x)
        # Act
        pha, amp, freqs = wavelet(xt, fs, device="auto")
        # Assert
        assert pha.device.type == "cuda"


class TestPacDeviceAuto:
    def test_pac_auto_values_is_ndarray(self):
        # Arrange
        x, fs = _pac_sig()
        # Act
        pac_values, pha_mids, amp_mids = pac(x, fs, device="auto")
        # Assert
        assert isinstance(pac_values, np.ndarray)

    def test_pac_auto_pha_mids_is_ndarray(self):
        # Arrange
        x, fs = _pac_sig()
        # Act
        pac_values, pha_mids, amp_mids = pac(x, fs, device="auto")
        # Assert
        assert isinstance(pha_mids, np.ndarray)

    def test_pac_auto_amp_mids_is_ndarray(self):
        # Arrange
        x, fs = _pac_sig()
        # Act
        pac_values, pha_mids, amp_mids = pac(x, fs, device="auto")
        # Assert
        assert isinstance(amp_mids, np.ndarray)

    def test_pac_auto_shape(self):
        # Arrange
        x, fs = _pac_sig()
        # Act
        pac_values, _, _ = pac(x, fs, device="auto")
        # Assert
        assert pac_values.shape == (1, 2, 100, 100)

    def test_pac_auto_matches_cpu_shape(self):
        # Arrange
        x, fs = _pac_sig()
        # Act
        pac_auto, _, _ = pac(x, fs)  # default device="auto"
        pac_cpu, _, _ = pac(x, fs, device="cpu")
        # Assert
        assert pac_auto.shape == pac_cpu.shape

    def test_pac_explicit_cpu_values_is_ndarray(self):
        # Arrange
        x, fs = _pac_sig()
        # Act
        pac_values, _, _ = pac(x, fs, device="cpu")
        # Assert
        assert isinstance(pac_values, np.ndarray)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_pac_auto_picks_cuda_when_available(self):
        # Arrange
        x, fs = _pac_sig()
        xt = torch.from_numpy(x)
        # Act
        pac_values, _, _ = pac(xt, fs, device="auto")
        # Assert
        assert pac_values.device.type == "cuda"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
