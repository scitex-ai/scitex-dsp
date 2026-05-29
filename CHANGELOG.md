# Changelog

All notable changes to `scitex-dsp` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.1.10]

- fix(deps): repoint `scitex_gen` imports to the public API (`from scitex_gen import to_z/to_even/to_odd`) after scitex-gen's `_numeric` reorg, which broke the old private `scitex_gen._norm`/`._to_even`/`._to_odd` paths against scitex-gen 0.1.10
- deps: raise floors to `scitex-gen>=0.1.10` and `scitex-nn>=0.1.13` (the versions carrying the fixes)
- test(gate): replace stale `scitex_gen._numeric._*` cross-package gate entries with the public `scitex_gen` (PS-140)

## [Unreleased]

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

### Added
- `examples/`: 13-notebook progressive tutorial gallery (`01_demo_sig`
  → `13_detect_ripples`) committed with executed cell outputs.
- `examples/00_run_all.sh` dispatcher and `examples/README.md` index.
- `tests/examples/test_<stem>.py` per notebook — each runs the file
  via `jupyter nbconvert --execute`.
- `README.md`: `## Architecture` (mermaid + src tree) and `## Demo`
  (mermaid tutorial-flow + gallery table) sections.

### Removed
- `examples/quickstart.py` and `tests/examples/test_quickstart.py` —
  superseded by the numbered notebook gallery.

## [0.1.6]

- Initial CHANGELOG entry — see git log for prior history.
