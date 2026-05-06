# scitex-dsp

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/assets/images/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center"><b>Digital signal processing for scientific Python — PAC, Hilbert, wavelet, filters, resampling, demo signals.</b></p>

<p align="center">
  <a href="https://scitex-dsp.readthedocs.io/">Full Documentation</a> · <code>pip install scitex-dsp</code>
</p>

<!-- scitex-badges:start -->
<p align="center">
  <a href="https://pypi.org/project/scitex-dsp/"><img src="https://img.shields.io/pypi/v/scitex-dsp.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/scitex-dsp/"><img src="https://img.shields.io/pypi/pyversions/scitex-dsp.svg" alt="Python"></a>
  <a href="https://github.com/ywatanabe1989/scitex-dsp/actions/workflows/test.yml"><img src="https://github.com/ywatanabe1989/scitex-dsp/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/ywatanabe1989/scitex-dsp"><img src="https://codecov.io/gh/ywatanabe1989/scitex-dsp/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scitex-dsp.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/scitex-dsp/badge/?version=latest" alt="Docs"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
</p>
<!-- scitex-badges:end -->

---

## Problem and Solution

| # | Problem | Solution |
|---|---------|----------|
| 1 | **Signal-processing pipelines mix NumPy, SciPy, MNE, and PyTorch with incompatible array shapes.** | **`scitex_dsp` exposes a uniform `(batch, channel, time)` 3-D contract via `ensure_3d` and the `@torch_fn` decorator.** |
| 2 | **Phase-Amplitude Coupling (PAC), wavelets, and ripple detection are scattered across one-off scripts.** | **First-class `pac`, `wavelet`, `hilbert`, `detect_ripples`, `modulation_index` reproducible primitives.** |
| 3 | **Demo signals for testing pipelines have to be re-rolled by every project.** | **`demo_sig(sig_type=...)` produces deterministic chirp / periodic / ripple test signals.** |

## Installation

```bash
pip install scitex-dsp
```

## 2 Interfaces

<details open>
<summary><strong>Python API</strong></summary>

<br>

```python
import scitex_dsp as dsp

xx, tt, fs = dsp.demo_sig(sig_type="chirp", fs=1024)
psd, ff = dsp.psd(xx, fs)
xf = dsp.filt.bandpass(xx, fs, bands=[[8, 12]])
hp = dsp.hilbert(xx)
pac, freqs_pha, freqs_amp = dsp.pac(xx, fs)
```

</details>

<details>
<summary><strong>Importable from the umbrella</strong></summary>

<br>

```python
import scitex
scitex.dsp.demo_sig(sig_type="chirp")  # `scitex.dsp` aliases `scitex_dsp`
```

</details>

## Quick Start

```python
import scitex_dsp

sig, t, fs = scitex_dsp.demo_sig(sig_type="periodic", batch_size=2, n_chs=4, t_sec=2, fs=256)
pp, ff = scitex_dsp.psd(sig, fs)
print("PSD shape:", pp.shape, "freqs:", ff[0], "->", ff[-1], "Hz")
```

## Part of SciTeX

`scitex-dsp` is part of [**SciTeX**](https://scitex.ai).

Install via the umbrella with `pip install scitex[dsp]`, then access as `scitex.dsp` or run `scitex dsp` from the CLI.

>Four Freedoms for Research
>
>0. The freedom to **run** your research anywhere — your machine, your terms.
>1. The freedom to **study** how every step works — from raw data to final manuscript.
>2. The freedom to **redistribute** your workflows, not just your papers.
>3. The freedom to **modify** any module and share improvements with the community.
>
>AGPL-3.0 — because we believe research infrastructure deserves the same freedoms as the software it runs on.

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/assets/images/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
</p>
