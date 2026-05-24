#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-01 21:00:00 (ywatanabe)"
# File: ./tests/scitex/dsp/test_add_noise.py

"""
Test module for scitex.dsp.add_noise functions.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np
from numpy.testing import assert_allclose


class TestAddNoise:
    """Test class for noise addition functions."""

    @pytest.fixture
    def clean_signal(self):
        """Create a clean test signal."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        return signal.astype(np.float32)

    @pytest.fixture
    def multi_channel_signal(self):
        """Create a multi-channel clean signal."""
        t = np.linspace(0, 1, 1000)
        n_channels = 4
        signals = []
        for i in range(n_channels):
            freq = 5 * (i + 1)  # 5, 10, 15, 20 Hz
            signals.append(np.sin(2 * np.pi * freq * t))
        return np.array(signals, dtype=np.float32)

    def test_module_import_hasattr_scitex_dsp_add_noise_gauss(self):
        # Arrange
        import scitex.dsp.add_noise
        # Act
        present = hasattr(scitex.dsp.add_noise, "gauss")
        # Assert
        assert present

    def test_module_import_hasattr_scitex_dsp_add_noise_white(self):
        # Arrange
        import scitex.dsp.add_noise
        # Act
        present = hasattr(scitex.dsp.add_noise, "white")
        # Assert
        assert present

    def test_module_import_hasattr_scitex_dsp_add_noise_pink(self):
        # Arrange
        import scitex.dsp.add_noise
        # Act
        present = hasattr(scitex.dsp.add_noise, "pink")
        # Assert
        assert present

    def test_module_import_hasattr_scitex_dsp_add_noise_brown(self):
        # Arrange
        import scitex.dsp.add_noise
        # Act
        present = hasattr(scitex.dsp.add_noise, "brown")
        # Assert
        assert present

    def test_gauss_noise_noisy_shape_equals_clean_signal_shape(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import gauss
        # Act
        noisy = gauss(clean_signal, amp=0.1)
        # Assert
        assert noisy.shape == clean_signal.shape

    def test_gauss_noise_not_np_array_equal_noisy_clean_signal(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import gauss
        # Act
        noisy = gauss(clean_signal, amp=0.1)
        # Assert
        assert not np.array_equal(noisy, clean_signal)

    def test_gauss_noise_zero_mean_property(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import gauss
        noisy = gauss(clean_signal, amp=0.1)
        # Act
        noise = noisy - clean_signal
        # Assert
        assert np.abs(np.mean(noise)) < 0.05  # Should be zero mean

    def test_gauss_noise_std_within_expected_range(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import gauss
        noisy = gauss(clean_signal, amp=0.1)
        # Act
        noise = noisy - clean_signal
        # Assert
        assert 0.05 < np.std(noise) < 0.15  # Should have std ~ amp

    def test_white_noise_noisy_shape_equals_clean_signal_shape(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import white
        amp = 0.2
        # Act
        noisy = white(clean_signal, amp=amp)
        # Assert
        assert noisy.shape == clean_signal.shape

    def test_white_noise_not_np_array_equal_noisy_clean_signal(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import white
        amp = 0.2
        # Act
        noisy = white(clean_signal, amp=amp)
        # Assert
        assert not np.array_equal(noisy, clean_signal)

    def test_white_noise_bounded_by_amplitude(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import white
        amp = 0.2
        noisy = white(clean_signal, amp=amp)
        # Act
        noise = noisy - clean_signal
        # Assert
        assert np.all(np.abs(noise) <= amp * 1.01)  # Small tolerance

    def test_pink_noise_noisy_shape_equals_clean_signal_shape(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import pink
        # Act
        noisy = pink(clean_signal, amp=0.1)
        # Assert
        assert noisy.shape == clean_signal.shape

    def test_pink_noise_not_np_array_equal_noisy_clean_signal(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import pink
        # Act
        noisy = pink(clean_signal, amp=0.1)
        # Assert
        assert not np.array_equal(noisy, clean_signal)

    def test_pink_noise_has_nonzero_std(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import pink
        noisy = pink(clean_signal, amp=0.1)
        # Act
        noise = noisy - clean_signal
        # Assert
        assert np.std(noise) > 0

    def test_brown_noise_noisy_shape_equals_clean_signal_shape(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import brown
        # Act
        noisy = brown(clean_signal, amp=0.1)
        # Assert
        assert noisy.shape == clean_signal.shape

    def test_brown_noise_not_np_array_equal_noisy_clean_signal(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import brown
        # Act
        noisy = brown(clean_signal, amp=0.1)
        # Assert
        assert not np.array_equal(noisy, clean_signal)

    def test_brown_noise_has_nonzero_std(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import brown
        noisy = brown(clean_signal, amp=0.1)
        # Act
        noise = noisy - clean_signal
        # Assert
        assert np.std(noise) > 0

    def test_amplitude_scaling_higher_amp_yields_larger_noise_std(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import gauss
        amp1, amp2 = 0.1, 0.5
        noisy1 = gauss(clean_signal, amp=amp1)
        noisy2 = gauss(clean_signal, amp=amp2)
        noise1 = noisy1 - clean_signal
        # Act
        noise2 = noisy2 - clean_signal
        # Assert
        assert np.std(noise2) > np.std(noise1)

    def test_amplitude_scaling_ratio_approximately_proportional(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import gauss
        amp1, amp2 = 0.1, 0.5
        noisy1 = gauss(clean_signal, amp=amp1)
        noisy2 = gauss(clean_signal, amp=amp2)
        noise1 = noisy1 - clean_signal
        noise2 = noisy2 - clean_signal
        # Act
        ratio = np.std(noise2) / np.std(noise1)
        # Assert
        assert 3 < ratio < 7  # Approximately amp2/amp1 = 5

    @pytest.mark.parametrize("noise_type", ["gauss", "white", "pink", "brown"])
    def test_multi_channel_noise_preserves_shape(self, multi_channel_signal, noise_type):
        # Arrange
        import scitex.dsp.add_noise as add_noise
        noise_func = getattr(add_noise, noise_type)
        # Act
        noisy = noise_func(multi_channel_signal, amp=0.1)
        # Assert
        assert noisy.shape == multi_channel_signal.shape

    @pytest.mark.parametrize("noise_type", ["gauss", "white", "pink", "brown"])
    def test_multi_channel_noise_channels_differ(self, multi_channel_signal, noise_type):
        # Arrange
        import scitex.dsp.add_noise as add_noise
        noise_func = getattr(add_noise, noise_type)
        noisy = noise_func(multi_channel_signal, amp=0.1)
        noise = noisy - multi_channel_signal
        # Act
        all_pairs_differ = all(
            not np.array_equal(noise[i], noise[i + 1])
            for i in range(len(noise) - 1)
        )
        # Assert
        assert all_pairs_differ

    @pytest.mark.parametrize("noise_type", ["gauss", "white", "pink", "brown"])
    def test_torch_tensor_input_returns_tensor(self, clean_signal, noise_type):
        # Arrange
        import scitex.dsp.add_noise as add_noise
        noise_func = getattr(add_noise, noise_type)
        signal_torch = torch.tensor(clean_signal)
        # Act
        noisy = noise_func(signal_torch, amp=0.1)
        # Assert
        assert isinstance(noisy, torch.Tensor)

    @pytest.mark.parametrize("noise_type", ["gauss", "white", "pink", "brown"])
    def test_torch_tensor_input_preserves_shape(self, clean_signal, noise_type):
        # Arrange
        import scitex.dsp.add_noise as add_noise
        noise_func = getattr(add_noise, noise_type)
        signal_torch = torch.tensor(clean_signal)
        # Act
        noisy = noise_func(signal_torch, amp=0.1)
        # Assert
        assert noisy.shape == signal_torch.shape

    @pytest.mark.parametrize("noise_type", ["gauss", "white", "pink", "brown"])
    def test_torch_tensor_input_adds_noise(self, clean_signal, noise_type):
        # Arrange
        import scitex.dsp.add_noise as add_noise
        noise_func = getattr(add_noise, noise_type)
        signal_torch = torch.tensor(clean_signal)
        # Act
        noisy = noise_func(signal_torch, amp=0.1)
        # Assert
        assert not torch.equal(noisy, signal_torch)

    def test_zero_amplitude_gauss_returns_unchanged(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import gauss
        # Act
        noisy_gauss = gauss(clean_signal, amp=0.0)
        # Assert
        assert np.allclose(noisy_gauss, clean_signal, rtol=1e-6)

    def test_zero_amplitude_white_returns_unchanged(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import white
        # Act
        noisy_white = white(clean_signal, amp=0.0)
        # Assert
        assert np.allclose(noisy_white, clean_signal, rtol=1e-6)

    def test_reproducibility_with_seed(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import gauss
        torch.manual_seed(42)
        noisy1 = gauss(torch.tensor(clean_signal), amp=0.1)
        torch.manual_seed(42)
        # Act
        noisy2 = gauss(torch.tensor(clean_signal), amp=0.1)
        # Assert
        assert torch.equal(noisy1, noisy2)

    def test_different_dimensions_1d_shape_preserved(self):
        # Arrange
        from scitex.dsp.add_noise import gauss
        signal_1d = torch.randn(1000)
        # Act
        noisy_1d = gauss(signal_1d, amp=0.1)
        # Assert
        assert noisy_1d.shape == signal_1d.shape

    def test_different_dimensions_2d_shape_preserved(self):
        # Arrange
        from scitex.dsp.add_noise import gauss
        signal_2d = torch.randn(4, 1000)
        # Act
        noisy_2d = gauss(signal_2d, amp=0.1)
        # Assert
        assert noisy_2d.shape == signal_2d.shape

    def test_different_dimensions_3d_shape_preserved(self):
        # Arrange
        from scitex.dsp.add_noise import gauss
        signal_3d = torch.randn(2, 4, 1000)
        # Act
        noisy_3d = gauss(signal_3d, amp=0.1)
        # Assert
        assert noisy_3d.shape == signal_3d.shape

    def test_pink_noise_spectrum_low_power_greater_than_high(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import pink
        long_signal = np.zeros(10000, dtype=np.float32)
        noisy = pink(long_signal, amp=1.0)
        noise = noisy - long_signal
        fft = np.abs(np.fft.rfft(noise))
        low_power = np.mean(fft[:100] ** 2)
        # Act
        high_power = np.mean(fft[-100:] ** 2)
        # Assert
        assert low_power > high_power

    def test_brown_noise_smoother_than_white(self, clean_signal):
        # Arrange
        from scitex.dsp.add_noise import brown
        noisy = brown(clean_signal, amp=0.1)
        noise = noisy - clean_signal
        # Act
        diff_noise = np.diff(noise)
        # Assert
        assert np.std(diff_noise) < np.std(noise)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    @pytest.mark.parametrize("noise_type", ["gauss", "white", "pink", "brown"])
    def test_device_handling_gpu_returns_cuda(self, noise_type):
        # Arrange
        import scitex.dsp.add_noise as add_noise
        noise_func = getattr(add_noise, noise_type)
        signal_gpu = torch.randn(1000).cuda()
        # Act
        noisy = noise_func(signal_gpu, amp=0.1)
        # Assert
        assert noisy.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    @pytest.mark.parametrize("noise_type", ["gauss", "white", "pink", "brown"])
    def test_device_handling_gpu_preserves_device(self, noise_type):
        # Arrange
        import scitex.dsp.add_noise as add_noise
        noise_func = getattr(add_noise, noise_type)
        signal_gpu = torch.randn(1000).cuda()
        # Act
        noisy = noise_func(signal_gpu, amp=0.1)
        # Assert
        assert noisy.device == signal_gpu.device

    @pytest.mark.parametrize(
        "noise_type,amp",
        [
            ("gauss", 0.1),
            ("gauss", 1.0),
            ("white", 0.5),
            ("pink", 0.2),
            ("brown", 0.1),
        ],
    )
    def test_signal_to_noise_ratio_in_reasonable_range(self, clean_signal, noise_type, amp):
        # Arrange
        import scitex.dsp.add_noise as add_noise
        noise_func = getattr(add_noise, noise_type)
        noisy = noise_func(clean_signal, amp=amp)
        signal_power = np.mean(clean_signal**2)
        noise = noisy - clean_signal
        noise_power = np.mean(noise**2)
        # Act
        snr_db = 10 * np.log10(signal_power / noise_power)
        # Assert
        assert -10 < snr_db < 40  # Reasonable SNR range


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/add_noise.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "ywatanabe (2024-11-02 23:09:49)"
# # File: ./scitex_repo/src/scitex/dsp/add_noise.py
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
#
# def _check_torch():
#     if not TORCH_AVAILABLE:
#         raise ImportError(
#             "PyTorch is not installed. Please install with: pip install torch"
#         )
#
#
# def _uniform(shape, amp=1.0):
#     _check_torch()
#     a, b = -amp, amp
#     return -amp + (2 * amp) * torch.rand(shape)
#
#
# @signal_fn
# def gauss(x, amp=1.0):
#     noise = amp * torch.randn(x.shape)
#     return x + noise.to(x.device)
#
#
# @signal_fn
# def white(x, amp=1.0):
#     return x + _uniform(x.shape, amp=amp).to(x.device)
#
#
# @signal_fn
# def pink(x, amp=1.0, dim=-1):
#     """
#     Adds pink noise to a given tensor along a specified dimension.
#
#     Parameters:
#     - x (torch.Tensor): The input tensor to which pink noise will be added.
#     - amp (float, optional): The amplitude of the pink noise. Defaults to 1.0.
#     - dim (int, optional): The dimension along which to add pink noise. Defaults to -1.
#
#     Returns:
#     - torch.Tensor: The input tensor with added pink noise.
#     """
#     cols = x.size(dim)
#     noise = torch.randn(cols, dtype=x.dtype, device=x.device)
#     noise = torch.fft.rfft(noise)
#     indices = torch.arange(1, noise.size(0), dtype=x.dtype, device=x.device)
#     noise[1:] /= torch.sqrt(indices)
#     noise = torch.fft.irfft(noise, n=cols)
#     noise = noise - noise.mean()
#     noise_amp = torch.sqrt(torch.mean(noise**2))
#     noise = noise * (amp / noise_amp)
#     return x + noise.to(x.device)
#
#
# @signal_fn
# def brown(x, amp=1.0, dim=-1):
#     from scitex.dsp import norm
#
#     noise = _uniform(x.shape, amp=amp)
#     noise = torch.cumsum(noise, dim=dim)
#     noise = norm.minmax(noise, amp=amp, dim=dim)
#     return x + noise.to(x.device)
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
#     T_SEC = 1
#     FS = 128
#
#     # Demo signal
#     xx, tt, fs = scitex.dsp.demo_sig(t_sec=T_SEC, fs=FS)
#
#     funcs = {
#         "orig": lambda x: x,
#         "gauss": gauss,
#         "white": white,
#         "pink": pink,
#         "brown": brown,
#     }
#
#     # Plots
#     fig, axes = scitex.plt.subplots(nrows=len(funcs), ncols=2, sharex=True, sharey=True)
#     count = 0
#     for (k, fn), axes_row in zip(funcs.items(), axes):
#         for ax in axes_row:
#             if count % 2 == 0:
#                 ax.plot(tt, fn(xx)[0, 0], label=k, c="blue")
#             else:
#                 ax.plot(tt, (fn(xx) - xx)[0, 0], label=f"{k} - orig", c="red")
#             count += 1
#             ax.legend(loc="upper right")
#
#     fig.supxlabel("Time [s]")
#     fig.supylabel("Amplitude [?V]")
#     axes[0, 0].set_title("Signal + Noise")
#     axes[0, 1].set_title("Noise")
#
#     scitex.io.save(fig, "traces.png")
#
#     # Close
#     scitex.session.close(CONFIG)
#
# # EOF
#
# """
# /home/ywatanabe/proj/entrance/scitex/dsp/add_noise.py
# """
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/add_noise.py
# --------------------------------------------------------------------------------
