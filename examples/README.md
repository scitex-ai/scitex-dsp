# scitex-dsp examples

Progressive Jupyter tutorials. Each notebook builds on the previous
and is committed with executed cell outputs — read them on GitHub
without running anything locally.

## Tutorial sequence

| #  | Notebook | What it covers |
|----|---|---|
| 01 | [`01_demo_sig.ipynb`](01_demo_sig.ipynb) | Synthesise test signals: uniform, gauss, periodic, chirp |
| 02 | [`02_shape_utils.ipynb`](02_shape_utils.ipynb) | `ensure_3d`, `crop` — shape helpers |
| 03 | [`03_norm.ipynb`](03_norm.ipynb) | `norm.z`, `norm.minmax` — per-trace normalization |
| 04 | [`04_filt.ipynb`](04_filt.ipynb) | `filt.bandpass` / `filt.bandstop` (Butterworth) |
| 05 | [`05_hilbert.ipynb`](05_hilbert.ipynb) | `hilbert` — analytic signal, vs `scipy.signal.hilbert` |
| 06 | [`06_psd.ipynb`](06_psd.ipynb) | `psd` + manual band-power integration |
| 07 | [`07_wavelet.ipynb`](07_wavelet.ipynb) | `wavelet` — continuous wavelet transform |
| 08 | [`08_resample.ipynb`](08_resample.ipynb) | `resample` — change sampling rate |
| 09 | [`09_add_noise.ipynb`](09_add_noise.ipynb) | `add_noise.{gauss,white,pink,brown}` + their PSDs |
| 10 | [`10_reference.ipynb`](10_reference.ipynb) | `reference.common_average` (CAR) |
| 11 | [`11_modulation_index.ipynb`](11_modulation_index.ipynb) | `modulation_index` — Tort MI from `bandpass + hilbert` |
| 12 | [`12_pac.ipynb`](12_pac.ipynb) | `pac` — full heatmap, side-by-side with `tensorpac` |
| 13 | [`13_detect_ripples.ipynb`](13_detect_ripples.ipynb) | `detect_ripples` — end-to-end ripple pipeline |

## Suggested reading paths

- **First-time user, 5 minutes** — `01_demo_sig` → `04_filt` → `05_hilbert` → `06_psd`.
- **Spectral analysis** — `04_filt` → `06_psd` → `07_wavelet` → `09_add_noise`.
- **Phase-amplitude coupling** — `04_filt` → `05_hilbert` → `11_modulation_index` → `12_pac`.
- **Hippocampal LFP** — `04_filt` → `10_reference` → `13_detect_ripples`.

## Re-running

The notebooks are committed with their cell outputs. To re-execute
all of them in place:

```bash
./00_run_all.sh
```

## Tested

Each notebook has a matching execution-smoke test in
[`../tests/examples/test_notebooks.py`](../tests/examples/test_notebooks.py)
that runs it via `jupyter nbconvert --execute` in a tmp dir. CI runs
this on every push.

## Conventions inside the notebooks

- `import scitex_dsp as dsp` — the package's standalone import.
- `%matplotlib inline` is placed **after** `import scitex_dsp` because
  importing the package switches the matplotlib backend.
- All signals follow the `(batch, channels, samples)` 3-D contract.
- Where applicable, results are cross-checked against the
  reference implementation (`scipy.signal`, `tensorpac`, etc.).
