---
description: |
  [TOPIC] Python API
  [DETAILS] Public Python API of scitex-dsp — exported functions, signatures,
  return types, and minimal usage examples per function.
tags: [scitex-dsp-python-api]
---

# Python API

## Signal generation

```python
xx, tt, fs = dsp.demo_sig(sig_type="chirp", t_sec=4.0, fs=1024)
```

`sig_type` supports: `"uniform"`, `"gauss"`, `"periodic"`, `"chirp"`, `"ripple"`, `"meg"`, `"tensorpac"`.

## Spectral analysis

```python
# Hilbert transform — analytic signal (phase + envelope)
phase, envelope = dsp.hilbert(x)

# Power spectral density
freqs, psd = dsp.psd(x, fs=1000)

# Per-band integrated power
band_powers = dsp.band_powers(x, fs=1000, bands={"theta": (4, 8), "gamma": (30, 80)})

# Continuous wavelet transform
coef, freqs = dsp.wavelet(x, fs=1000)

# Phase-amplitude coupling
mi = dsp.modulation_index(x, lo=(4, 8), hi=(30, 80), fs=1000)
pac_score = dsp.pac(x, lo=(4, 8), hi=(30, 80), fs=1000)
```

## Ripple detection

```python
ripples = dsp.detect_ripples(x, fs=1000, lo=150, hi=250, duration_min=0.03, duration_max=0.2)
# Returns DataFrame with columns: start_idx, end_idx, peak_idx, amplitude, energy
```

## Filtering

```python
# Butterworth bandpass/bandstop — pass bands as [[low, high], ...]
y_bp = dsp.filt.bandpass(x, fs=1000, bands=[[4, 8], [13, 30]])
y_bs = dsp.filt.bandstop(x, fs=1000, bands=[[48, 52]])

# Gaussian filter (smoothing)
y_g = dsp.filt.gauss(x, sigma=10)
```

## Normalization

```python
z = dsp.norm.z(x, dim=-1)           # z-score along time
mm = dsp.norm.minmax(x, dim=-1)         # min-max scale to [0, 1]
```

## Referencing

```python
car = dsp.reference.common_average(x, dim=-2)   # common-average reference
bip = dsp.reference.random(x, dim=-2)            # random-reference subtraction
ref = dsp.reference.take_reference(x, tgt_indi, dim=-2)  # subtract target channel
```

## Pre/post-processing

```python
x_3d = dsp.ensure_3d(x)                        # promote (C, T) -> (1, C, T)
x_cropped = dsp.crop(x, start=0.0, end=2.0, fs=1000)
x_rs = dsp.resample(x, old_fs=1000, new_fs=256)
noisy = dsp.add_noise.gauss(x, std=0.1)
segments = dsp.to_segments(x, window=256, stride=128)
```

## Noise injection

```python
dsp.add_noise.gauss(x)   # Gaussian noise
dsp.add_noise.white(x)   # white noise (flat PSD)
dsp.add_noise.pink(x)    # pink noise (1/f PSD)
dsp.add_noise.brown(x)   # brown noise (1/f² PSD)
```

## Utility submodules

- `dsp.filt` — Butterworth and Gaussian filters
- `dsp.norm` — z-score, min-max, robust normalization
- `dsp.reference` — common-average, random, bipolar referencing
- `dsp.add_noise` — noise generation (gauss, white, pink, brown)
- `dsp.params` — canonical frequency band definitions (delta, theta, alpha, beta, gamma)
- `dsp.example` — worked example pipelines

All functions accept `(channels, samples)` or `(batch, channels, samples)` arrays of type `numpy.ndarray` or `torch.Tensor`. Outputs preserve input shape and type.
