# scitex-dsp

Digital signal processing utilities (PAC, Hilbert, Wavelet, filters, resampling, demo signals, …) extracted from the [SciTeX](https://github.com/ywatanabe1989/scitex-python) ecosystem as a standalone package.

## Install

```bash
pip install scitex-dsp
```

## API

```python
import scitex_dsp as dsp

xx, tt, fs = dsp.demo_sig(sig_type="chirp", fs=1024)
psd, ff = dsp.psd(xx, fs)
xf = dsp.filt.bandpass(xx, fs, bands=[[8, 12]])
hp = dsp.hilbert(xx)
pac, freqs_pha, freqs_amp = dsp.pac(xx, fs, ...)
```

## Status

Standalone fork of `scitex.dsp`. The umbrella package's `scitex.dsp` import path
is preserved via a `sys.modules`-alias bridge.

Decoupling notes:
- `scitex.{decorators,gen,nn}` → `scitex_*` direct imports.
- `scitex.io.load_configs` deferred via `try/except` with `CONFIG = {}` fallback
  (only used in `_demo_sig.py` module-init).
- `scitex.dsp.utils` and `scitex.dsp._demo_sig` are intra-package — rewritten as
  `scitex_dsp.utils` / `scitex_dsp._demo_sig`.
- Example `if __name__ == "__main__"` blocks still reference the umbrella
  (`scitex.session`, `scitex.io`, `scitex.plt`) — they only run when the
  umbrella is installed.

## License

AGPL-3.0-only.
