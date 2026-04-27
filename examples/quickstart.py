"""scitex-dsp quickstart: build a demo signal and compute its PSD."""

import numpy as np

import scitex_dsp


def main():
    # 1. demo_sig: synthesize a multi-channel periodic test signal of shape
    # (batch, channels, time).
    sig, t, fs = scitex_dsp.demo_sig(
        sig_type="periodic",
        batch_size=2,
        n_chs=4,
        t_sec=2,
        fs=256,
    )
    print(
        "signal shape:", sig.shape, "| fs =", fs, "Hz | duration =", t[-1] - t[0], "s"
    )
    assert sig.ndim == 3
    assert sig.shape == (2, 4, int(2 * fs))

    # 2. psd: power spectral density via FFT, returned as (power, freqs).
    pp, ff = scitex_dsp.psd(sig, fs)
    print("psd power shape:", tuple(pp.shape))
    print("psd freq range:", float(ff.min()), "->", float(ff.max()), "Hz")
    assert pp.shape[:-1] == sig.shape[:-1]
    assert ff.min() >= 0
    assert ff.max() <= fs / 2 + 1e-6

    # 3. resample: downsample from 256 Hz to 128 Hz.
    down = scitex_dsp.resample(sig, src_fs=fs, tgt_fs=fs // 2)
    print("\nresampled to", fs // 2, "Hz, shape:", tuple(down.shape))
    assert down.shape[-1] == sig.shape[-1] // 2

    # 4. ensure_3d: make a 1-D / 2-D array conform to (batch, channel, time).
    flat = np.random.randn(512)
    promoted = scitex_dsp.ensure_3d(flat)
    print("ensure_3d(1D) shape:", tuple(promoted.shape))
    assert promoted.ndim == 3


if __name__ == "__main__":
    main()
