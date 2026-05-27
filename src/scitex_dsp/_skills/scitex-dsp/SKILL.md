---
name: scitex-dsp
description: Digital signal-processing utilities tuned for neuroscience workflows. `hilbert(x)` returns analytic signal; `psd(x, fs, ...)` and `band_powers(x, fs, bands)` compute spectral density. `pac(x, lo, hi, ...)` and `modulation_index(x, ...)` quantify phase-amplitude coupling. `detect_ripples(x, fs, ...)` runs the Buzsaki-style ripple detector with edge handling, find-events, and column sorting baked in. Pre-/post-processing: `crop`, `ensure_3d`, `resample`, `demo_sig` (synthetic test signal). Submodules: `filt` (Butterworth bandpass/bandstop), `norm` (z-score, min-max, robust), `reference` (CAR, bipolar, Laplacian), `add_noise` (white, pink, ar1), `params` (canonical band definitions), `example` (worked examples). All functions accept `(channels, samples)` or `(batch, channels, samples)`. Drop-in replacement for hand-rolled `scipy.signal.butter` + `lfilter` chains and bespoke ripple detectors.
primary_interface: python
interfaces:
  python: 3
  cli: 0
  mcp: 0
  skills: 2
  hook: 0
  http: 0
canonical-location: scitex-dsp/src/scitex_dsp/_skills/scitex-dsp/SKILL.md
tags: [scitex-dsp, scitex-package, signal-processing, neuroscience, ripples]
version: 0.1.6
exported_via: installed
---

# scitex-dsp

Digital signal-processing helpers for neuroscience-shaped data
(`(channels, samples)` or `(batch, channels, samples)`).

## Spectral

```python
import scitex_dsp as dsp

ana = dsp.hilbert(x)                       # analytic signal
freqs, p = dsp.psd(x, fs=1000)             # power spectral density
band_p = dsp.band_powers(x, fs=1000,
                         bands={"theta": (4, 8), "gamma": (30, 80)})
```

## Phase-amplitude coupling

```python
mi = dsp.modulation_index(x, lo=(4, 8), hi=(30, 80), fs=1000)
pac_score = dsp.pac(x, lo=(4, 8), hi=(30, 80), fs=1000)

# Cross-check against tensorpac (verification helper, not main use):
from scitex_dsp.utils import _calc_pac_with_tensorpac      # direct callable
from scitex_dsp.utils import pac as _pac_compare           # full submodule
```

Three pac surfaces are public:

| Path | Purpose |
| --- | --- |
| `dsp.pac(x, fs, ...)` | main heatmap function (core science) |
| `dsp.utils.pac` | cross-check submodule (calc + plot vs `tensorpac.Pac`) |
| `dsp.utils._calc_pac_with_tensorpac` | underscore alias — direct callable |

## Ripple detection

```python
ripples = dsp.detect_ripples(x, fs=1000, lo=150, hi=250, ...)
# Returns DataFrame with start/end/peak indices and edge handling.
```

## Filters / normalization / referencing

```python
y = dsp.filt.bandpass(x, lo=4, hi=8, fs=1000)
z = dsp.norm.zscore(x, axis=-1)
ref = dsp.reference.car(x)                  # common-average reference
```

## Pre-processing helpers

- `dsp.crop(x, start, end, fs)` — time-window a signal
- `dsp.ensure_3d(x)` — promote `(C,T)` to `(1,C,T)` for batched APIs
- `dsp.resample(x, old_fs, new_fs)` — polyphase resampling
- `dsp.demo_sig(...)` — synthetic test signal with known PSD

## Submodules

- `dsp.filt`, `dsp.norm`, `dsp.reference`, `dsp.add_noise`,
  `dsp.params`, `dsp.example` — see each submodule for details.

## When to use

- ✅ Neuroscience-shaped multi-channel time series (LFP, EEG, MEG, ECoG)
- ✅ Pipelines that need ripple/PAC/PSD as first-class operations
- ❌ Trainable / differentiable filters — use `scitex-nn` instead
- ❌ Single-channel scalar time series — `scipy.signal` is fine

## See also

- `scitex-nn` — trainable PyTorch counterparts of these primitives
- `scitex-stats` — statistical tests on derived spectral metrics
- `scitex-plt` / `figrecipe` — publication-ready spectrograms

## Sub-skills

### Core (01–09)
- [01_installation.md](01_installation.md) — install + import sanity check
- [02_quick-start.md](02_quick-start.md) — 30-second tour
- [03_python-api.md](03_python-api.md) — Python API surface
