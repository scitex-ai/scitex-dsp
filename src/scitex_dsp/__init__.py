#!/usr/bin/env python3
"""scitex-dsp — Digital signal-processing primitives for neuroscience.

Functionalities
---------------
- `hilbert(x)` — analytic signal (phase + envelope).
- `psd(x, fs, ...)` / `band_powers(x, fs, bands)` — power spectral
  density and per-band integrated power.
- `pac(x, lo, hi, ...)` / `modulation_index(x, ...)` — phase-amplitude
  coupling and the canonical MI estimator.
- `wavelet(...)` — continuous wavelet transform.
- `pac_features(pac_z, phase=None, ...)` — canonical PAC-summary
  feature set returning a flat `{names, values}` descriptor, plus
  `PAC_FEATURE_REGISTRY` / `feature_registry()` for provenance.
- `extract_all(x, fs=None, sets=..., ...)` — multi-backend feature
  facade emitting a single flat `{names, values}` vector over many
  engines (`"pac"` delegate + `"catch22"` wrap), with
  `extract_all_registry()` / `AVAILABLE_BACKENDS` for provenance.
- `detect_ripples(x, fs, ...)` — Buzsaki-style ripple detector with
  edge handling, find-events, and column sorting.
- `demo_sig(sig_type=...)` — deterministic chirp / periodic / ripple
  / noise test signals.
- `crop`, `ensure_3d`, `resample`, `time`, `add_noise`,
  `to_segments`, `to_sktime_df` — pre-/post-processing utilities.
- Submodules: `filt` (Butterworth bandpass/bandstop), `norm`
  (z-score / min-max / robust), `reference` (CAR / bipolar /
  Laplacian), `params` (canonical band definitions), `example`
  (worked examples).

IO
--
- Reads: numeric arrays (`numpy.ndarray`, `torch.Tensor`,
  `pandas.DataFrame`) in `(channels, samples)` or
  `(batch, channels, samples)` shape; optional audio devices via
  PortAudio; optional EEG positions via MNE.
- Writes: nothing by default — pure functions returning new arrays.
  Audio playback via `_audio_io._listen` when PortAudio is present.

Dependencies
------------
- Hard: `numpy`, `scipy`, `torch`, `pandas`, `mne`.
- Optional: `sounddevice` + PortAudio (audio I/O), `tensorpac`
  (cross-check for PAC).

Standalone import::

    import scitex_dsp as dsp
    xx, tt, fs = dsp.demo_sig(sig_type="chirp", fs=1024)
    ana = dsp.hilbert(xx)

Performance
-----------
All heavy dependencies (`torch`, `pandas`, `mne`, `matplotlib`,
`scitex_nn`, `scitex_gen`) are imported lazily on first attribute
access via PEP 562 ``__getattr__``. ``import scitex_dsp`` therefore
costs only a few milliseconds and does *not* pull in torch.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v

    try:
        __version__ = _v("scitex-dsp")
    except PackageNotFoundError:
        __version__ = "0.0.0+local"
    del _v, PackageNotFoundError
except ImportError:  # pragma: no cover — only on ancient Pythons
    __version__ = "0.0.0+local"


# ---------------------------------------------------------------------------
# PEP 562 lazy attribute map: public-name → submodule (relative).
# Keep the LHS == the public symbol, the RHS == the submodule that defines it.
# Importing any of these is what pulls in torch/pandas/mne — so we defer the
# import to first access instead of paying it at ``import scitex_dsp``.
# ---------------------------------------------------------------------------
_LAZY_ATTRS: dict[str, str] = {
    # Submodules re-exported at the top level.
    "example": ".example",
    "filt": ".filt",
    "norm": ".norm",
    "params": ".params",
    "reference": ".reference",
    "add_noise": "._synthesis.add_noise",
    # Spectral primitives.
    "hilbert": "._spectral",
    "psd": "._spectral",
    "band_powers": "._spectral",
    "pac": "._spectral",
    "wavelet": "._spectral",
    "modulation_index": "._spectral",
    "_reshape": "._spectral",
    # Feature-extraction primitives (named scalar-feature vectors).
    "pac_features": "._features",
    "feature_registry": "._features",
    "PAC_FEATURE_REGISTRY": "._features",
    # Multi-backend feature facade (single flat named vector over engines).
    "extract_all": "._features",
    "extract_all_registry": "._features",
    "AVAILABLE_BACKENDS": "._features",
    # Pre-/post-processing utilities.
    "crop": "._crop",
    "ensure_3d": "._ensure_3d",
    "resample": "._resample",
    "time": "._time",
    "demo_sig": "._synthesis",
    "to_segments": "._transform",
    "to_sktime_df": "._transform",
    "detect_ripples": "._detect_ripples",
    # Ripple-detection internals kept in the public namespace for
    # backwards compatibility (downstream/tests import them by name).
    "_calc_relative_peak_position": "._detect_ripples",
    "_drop_ripples_at_edges": "._detect_ripples",
    "_find_events": "._detect_ripples",
    "_preprocess": "._detect_ripples",
    "_sort_columns": "._detect_ripples",
}

# Optional public names that may not be importable (need PortAudio / MNE).
# Resolve once, lazily; ``None`` if the optional dependency is missing.
_OPTIONAL_ATTRS: dict[str, tuple[str, str]] = {
    "list_and_select_device": ("._audio_io._listen", "list_and_select_device"),
    "get_eeg_pos": ("._audio_io._mne", "get_eeg_pos"),
}

# ---------------------------------------------------------------------------
# Backwards-compatibility submodule aliases — files moved into topical
# subpackages (PS108b refactor) but downstream code may still import them by
# the old flat path. We register the moved submodules under their pre-refactor
# names *lazily* via a meta-path finder so that ``import scitex_dsp._hilbert``
# keeps working without eagerly importing torch at ``import scitex_dsp`` time.
# name (full module path) → relative target module that actually defines it.
# ---------------------------------------------------------------------------
_BC_ALIASES: dict[str, str] = {
    # NOTE: ``scitex_dsp.add_noise`` is now a *physical* re-export module
    # (src/scitex_dsp/add_noise.py), so it resolves via normal import
    # machinery under both ``scitex_dsp.add_noise`` and the umbrella alias
    # ``scitex.dsp.add_noise``; no meta-path entry is needed here.
    "scitex_dsp._demo_sig": "._synthesis._demo_sig",
    "scitex_dsp._hilbert": "._spectral._hilbert",
    "scitex_dsp._modulation_index": "._spectral._modulation_index",
    "scitex_dsp._pac": "._spectral._pac",
    "scitex_dsp._psd": "._spectral._psd",
    "scitex_dsp._wavelet": "._spectral._wavelet",
    "scitex_dsp._listen": "._audio_io._listen",
    "scitex_dsp._mne": "._audio_io._mne",
}


def _load_lazy_attr(name: str):
    """Resolve a `_LAZY_ATTRS` name, cache it in globals, and return it."""
    from importlib import import_module

    mod_name = _LAZY_ATTRS.get(name)
    if mod_name is None:
        return None
    mod = import_module(mod_name, __name__)
    # Submodules whose RHS *is* the target (e.g. ``add_noise`` →
    # ``._synthesis.add_noise``, ``filt`` → ``.filt``) resolve to the module
    # object itself; everything else is an attribute on the imported module.
    attr = mod if mod.__name__.rsplit(".", 1)[-1] == name else getattr(mod, name)
    globals()[name] = attr
    return attr


def _load_optional_attr(name: str):
    """Resolve an `_OPTIONAL_ATTRS` name and cache it (None on failure)."""
    from importlib import import_module

    spec = _OPTIONAL_ATTRS.get(name)
    if spec is None:
        return None
    mod_name, attr_name = spec
    try:
        mod = import_module(mod_name, __name__)
        attr = getattr(mod, attr_name, None)
    except (ImportError, OSError):
        attr = None
    globals()[name] = attr
    return attr


def __getattr__(name: str):
    """PEP 562 lazy-loader: import on first access, cache, return."""
    if name in _LAZY_ATTRS:
        return _load_lazy_attr(name)
    if name in _OPTIONAL_ATTRS:
        return _load_optional_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(_LAZY_ATTRS) | set(_OPTIONAL_ATTRS) | set(globals()))


# ---------------------------------------------------------------------------
# Lazy backwards-compat alias finder. Installing a light meta-path finder lets
# ``import scitex_dsp._hilbert`` (and friends) succeed by transparently
# importing the relocated module, *without* importing anything heavy at
# ``import scitex_dsp`` time.
# ---------------------------------------------------------------------------
class _BCAliasFinder:
    """importlib meta-path finder for relocated backward-compat modules."""

    def find_module(self, fullname, path=None):  # legacy API (py<3.4 compat)
        return self if fullname in _BC_ALIASES else None

    def load_module(self, fullname):  # pragma: no cover — legacy path
        import sys as _sys

        if fullname in _sys.modules:
            return _sys.modules[fullname]
        return self._import(fullname)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _BC_ALIASES:
            return None
        from importlib.machinery import ModuleSpec

        return ModuleSpec(fullname, self)

    def create_module(self, spec):
        return self._import(spec.name)

    def exec_module(self, module):
        # Module is already fully initialised (it's the real target module);
        # nothing further to execute.
        return None

    @staticmethod
    def _import(fullname):
        import sys as _sys
        from importlib import import_module

        target = import_module(_BC_ALIASES[fullname], __name__)
        _sys.modules[fullname] = target
        return target


def _install_bc_finder() -> None:
    import sys as _sys

    if not any(isinstance(f, _BCAliasFinder) for f in _sys.meta_path):
        _sys.meta_path.append(_BCAliasFinder())


_install_bc_finder()


# Public API: every non-underscore lazy/optional name (underscore-prefixed
# entries are backwards-compat internals that stay importable but private).
__all__ = [
    "__version__",
    *(n for n in _LAZY_ATTRS if not n.startswith("_")),
    *(n for n in _OPTIONAL_ATTRS if not n.startswith("_")),
]
