# Changelog

All notable changes to `scitex-dsp` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-07-02

### Added
- CPU-safe `device="auto"` default in `pac()` / `wavelet()` (resolved via
  `_spectral/_device.py`) — picks CUDA when available, else CPU, so the same
  call runs unchanged on a laptop or a GPU host.
- `pac_features()` — canonical PAC-summary feature set returning a flat
  `{names, values}` descriptor, plus a self-documenting feature registry
  (`PAC_FEATURE_REGISTRY` / `feature_registry()`) for provenance.
- `extract_all(x, fs) -> {names, values}` — multi-backend feature front-end
  emitting a single flat named vector over many engines (`pac` + `catch22`),
  with `extract_all_registry()` / `AVAILABLE_BACKENDS` for provenance.
- Optional `gpac` GPU-PAC backend for `pac(backend="gpac")` (the third-party
  `gpu-pac` engine); `backend="auto"` keeps the default CPU-safe path.
- Registry-driven feature redundancy-audit helpers: `feature_correlation`,
  `redundancy_summary`, `pca_loadings`, and `correlation_by_group`.
- `examples/`: 13-notebook progressive tutorial gallery (`01_demo_sig` →
  `13_detect_ripples`) committed with executed cell outputs, an
  `examples/00_run_all.sh` dispatcher, an `examples/README.md` index, and
  per-notebook `tests/examples/test_<stem>.py` runners.
- `README.md`: `## Architecture` (mermaid + src tree) and `## Demo`
  (mermaid tutorial-flow + gallery table) sections.

### Changed
- Raised `requires-python` to `>=3.11` (was `>=3.9`) to match the tested
  matrix (3.11/3.12/3.13) and the `scitex-stats>=0.2.0` core-dep floor, so pip
  fails early and clearly on unsupported interpreters instead of late in
  dependency resolution.
- Consolidated the optional-dependency extras into a single `[all]`
  (`tensorpac`, `sounddevice`, `pycatch22`, `scikit-learn`, `gpu-pac>=0.3.4`);
  the old `[audio]` / `[pac]` / `[catch22]` / `[features]` extras were removed.
- The `gpac` GPU-PAC backend now installs from PyPI via the `gpu-pac`
  distribution (`pip install scitex-dsp[all]`, or `pip install gpu-pac`) —
  pinned as a plain `gpu-pac>=0.3.4` specifier, never a direct git URL.

### Fixed
- `detect_ripples`: three latent bugs that produced 0 detections.
  - Nyquist violation in the internal downsample (`fs_tgt = low_hz*3`
    placed bandpass upper edge above Nyquist when high_hz > 1.5×low_hz).
    Now `max(low*3, ⌈high*2.5⌉)` with a warning when the safety floor
    kicks in.
  - Single-channel demean wiped the signal (cross-channel mean equals
    the signal). Now skipped with a warning when `n_chs == 1`.
  - Event-boundary threshold was `< 0` on the z-scored envelope, which
    yielded 0.7s "ripples" that swallowed neighbours. Now `< 0.5*sd`,
    giving physiological ~100 ms widths.

### Removed
- `examples/quickstart.py` and `tests/examples/test_quickstart.py` —
  superseded by the numbered notebook gallery.

## [0.1.10]

- fix(deps): repoint `scitex_gen` imports to the public API (`from scitex_gen import to_z/to_even/to_odd`) after scitex-gen's `_numeric` reorg, which broke the old private `scitex_gen._norm`/`._to_even`/`._to_odd` paths against scitex-gen 0.1.10
- deps: raise floors to `scitex-gen>=0.1.10` and `scitex-nn>=0.1.13` (the versions carrying the fixes)
- test(gate): replace stale `scitex_gen._numeric._*` cross-package gate entries with the public `scitex_gen` (PS-140)

## [0.1.6]

- Initial CHANGELOG entry — see git log for prior history.
