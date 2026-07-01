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
    def test_wavelet_auto_runs_on_cpu(self):
        """wavelet(..., device="auto") returns 3 arrays on a CPU-only box."""
        x, fs = _wavelet_sig()
        pha, amp, freqs = wavelet(x, fs, device="auto")
        assert isinstance(pha, np.ndarray)
        assert isinstance(amp, np.ndarray)
        assert isinstance(freqs, np.ndarray)

    def test_wavelet_auto_batch_dim_preserved(self):
        x, fs = _wavelet_sig()
        pha, amp, freqs = wavelet(x, fs, device="auto")
        assert pha.shape[0] == 1  # batch size

    def test_wavelet_auto_default_matches_explicit_cpu(self):
        """The default (auto) and explicit cpu produce the same shapes."""
        x, fs = _wavelet_sig()
        pha_auto, amp_auto, _ = wavelet(x, fs)  # default device="auto"
        pha_cpu, amp_cpu, _ = wavelet(x, fs, device="cpu")
        assert pha_auto.shape == pha_cpu.shape
        assert amp_auto.shape == amp_cpu.shape

    def test_wavelet_explicit_cpu_still_works(self):
        x, fs = _wavelet_sig()
        pha, amp, freqs = wavelet(x, fs, device="cpu")
        assert isinstance(amp, np.ndarray)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_wavelet_auto_picks_cuda_when_available(self):
        x, fs = _wavelet_sig()
        xt = torch.from_numpy(x)
        pha, amp, freqs = wavelet(xt, fs, device="auto")
        # Output device follows the resolved compute device (cuda here).
        assert pha.device.type == "cuda"


class TestPacDeviceAuto:
    def test_pac_auto_runs_on_cpu(self):
        """pac(..., device="auto") returns PAC values on a CPU-only box."""
        x, fs = _pac_sig()
        pac_values, pha_mids, amp_mids = pac(x, fs, device="auto")
        assert isinstance(pac_values, np.ndarray)
        assert isinstance(pha_mids, np.ndarray)
        assert isinstance(amp_mids, np.ndarray)

    def test_pac_auto_shape(self):
        x, fs = _pac_sig()
        pac_values, _, _ = pac(x, fs, device="auto")
        # (batch, n_chs, pha_n_bands, amp_n_bands) with defaults 100 x 100.
        assert pac_values.shape == (1, 2, 100, 100)

    def test_pac_auto_default_matches_explicit_cpu(self):
        x, fs = _pac_sig()
        pac_auto, _, _ = pac(x, fs)  # default device="auto"
        pac_cpu, _, _ = pac(x, fs, device="cpu")
        assert pac_auto.shape == pac_cpu.shape

    def test_pac_explicit_cpu_still_works(self):
        x, fs = _pac_sig()
        pac_values, _, _ = pac(x, fs, device="cpu")
        assert isinstance(pac_values, np.ndarray)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_pac_auto_picks_cuda_when_available(self):
        x, fs = _pac_sig()
        xt = torch.from_numpy(x)
        pac_values, _, _ = pac(xt, fs, device="auto")
        assert pac_values.device.type == "cuda"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
