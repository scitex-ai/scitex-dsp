---
description: |
  [TOPIC] Quick Start
  [DETAILS] Smallest useful example demonstrating the primary use case in
  under 30 seconds.
tags: [scitex-dsp-quick-start]
---

# Quick Start

```python
import scitex_dsp as dsp

# Generate a synthetic chirp signal (1 batch, 1 channel, 1024 samples)
xx, tt, fs = dsp.demo_sig(sig_type="chirp", fs=1024)

# Compute power spectral density
freqs, psd = dsp.psd(xx, fs)

# Filter in the theta band (4-8 Hz)
filtered = dsp.filt.bandpass(xx, fs, bands=[[4, 8]])

# Compute analytic signal (phase + envelope)
phase, envelope = dsp.hilbert(xx)

# Detect ripples (Buzsaki-style)
ripples = dsp.detect_ripples(xx, fs, lo=150, hi=250)
print(ripples)
```
