#!/usr/bin/env python3
"""Scitex dsp module."""

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

# Backwards-compatibility submodule aliases — files moved into
# topical subpackages (PS108b refactor) but downstream code still
# imports them by the old flat path. Register the moved submodules
# under their pre-refactor names so `import scitex_dsp.add_noise`,
# `from scitex_dsp._hilbert import …` etc. keep working.
import sys as _sys
import warnings

# Submodules: example/params/norm/reference/filt at root,
# add_noise re-exported from _synthesis for backwards compatibility.
from . import example, filt, norm, params, reference

# Optional audio submodules — wrapped because _listen imports
# sounddevice (needs PortAudio) and _mne imports mne.
try:
    from ._audio_io import _listen as _bc_listen
except (ImportError, OSError):
    _bc_listen = None
try:
    from ._audio_io import _mne as _bc_mne
except Exception:  # mne import path may raise misc runtime errors
    _bc_mne = None

# Core imports that should always work
from ._crop import crop
from ._detect_ripples import (
    _calc_relative_peak_position,
    _drop_ripples_at_edges,
    _find_events,
    _preprocess,
    _sort_columns,
    detect_ripples,
)
from ._ensure_3d import ensure_3d
from ._resample import resample
from ._spectral import (
    _hilbert as _bc_hilbert,
)
from ._spectral import (
    _modulation_index as _bc_modulation_index,
)
from ._spectral import (
    _pac as _bc_pac,
)
from ._spectral import (
    _psd as _bc_psd,
)
from ._spectral import (
    _reshape,
    band_powers,
    hilbert,
    modulation_index,
    pac,
    psd,
    wavelet,
)
from ._spectral import (
    _wavelet as _bc_wavelet,
)
from ._synthesis import _demo_sig as _bc_demo_sig
from ._synthesis import add_noise, demo_sig
from ._time import time
from ._transform import to_segments, to_sktime_df

for _old, _mod in {
    "scitex_dsp.add_noise": add_noise,
    "scitex_dsp._demo_sig": _bc_demo_sig,
    "scitex_dsp._hilbert": _bc_hilbert,
    "scitex_dsp._listen": _bc_listen,
    "scitex_dsp._mne": _bc_mne,
    "scitex_dsp._modulation_index": _bc_modulation_index,
    "scitex_dsp._pac": _bc_pac,
    "scitex_dsp._psd": _bc_psd,
    "scitex_dsp._wavelet": _bc_wavelet,
}.items():
    _sys.modules.setdefault(_old, _mod)
del _sys

# Try to import audio-related functions that require PortAudio
try:
    from ._audio_io._listen import list_and_select_device

    _audio_available = True
except (ImportError, OSError):
    warnings.warn(
        "Audio functionality unavailable: PortAudio library not found. "
        "Install PortAudio to use audio features (e.g., sudo apt-get install portaudio19-dev)",
        ImportWarning,
    )
    list_and_select_device = None
    _audio_available = False

# Try to import MNE-related functions
try:
    from ._audio_io._mne import get_eeg_pos

    _mne_available = True
except ImportError:
    warnings.warn(
        "MNE functionality unavailable. Install MNE-Python to use EEG position features.",
        ImportWarning,
    )
    get_eeg_pos = None
    _mne_available = False

__all__ = [
    "__version__",
    "add_noise",
    "band_powers",
    "crop",
    "demo_sig",
    "detect_ripples",
    "ensure_3d",
    "example",
    "filt",
    "get_eeg_pos",
    "hilbert",
    "list_and_select_device",
    "modulation_index",
    "norm",
    "pac",
    "params",
    "psd",
    "reference",
    "resample",
    "time",
    "to_segments",
    "to_sktime_df",
    "wavelet",
]
