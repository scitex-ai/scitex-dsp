"""End-to-end: full DSP pipeline on a real synthetic signal.

Why an e2e layer for this: the unit tests in this repo prove the PIECES
(demo-signal synthesis, bandpass filtering, Hilbert transform). What they
cannot prove is that the pieces compose into the journey users actually
run — synthesize → filter → instantaneous phase/amplitude — without a
shape, dtype, or finiteness break at a stage boundary.

Gated like every slow layer: needs `RUN_E2E=1` (so the default suite stays
fast); skipped by default. Loopback only — no network, no hardware.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.e2e

_RUN_E2E = os.environ.get("RUN_E2E") == "1"
needs_e2e = pytest.mark.skipif(not _RUN_E2E, reason="e2e layer needs RUN_E2E=1")


def _run_pipeline():
    """Synthesize a chirp, bandpass it, then Hilbert-transform it."""
    # Arrange
    import scitex_dsp as dsp

    # Act
    xx, _tt, fs = dsp.demo_sig(t_sec=2.0, fs=256, sig_type="chirp")
    filt_xx = dsp.filt.bandpass(xx, fs, [[20, 50]])
    pha, amp = dsp.hilbert(filt_xx)
    # Assert
    assert xx is not None
    return xx, filt_xx, pha, amp


@needs_e2e
def test_pipeline_filtered_shape_matches_input():
    # Arrange
    xx, filt_xx, _pha, _amp = _run_pipeline()
    # Act
    result = filt_xx.shape
    # Assert
    assert result == (xx.shape[0], xx.shape[1], 1, xx.shape[2])


@needs_e2e
def test_pipeline_hilbert_amplitude_finite():
    # Arrange
    _xx, _filt_xx, _pha, amp = _run_pipeline()
    # Act
    result = np.asarray(amp)
    # Assert
    assert np.isfinite(result).all()


@needs_e2e
def test_pipeline_hilbert_phase_shape_matches_input():
    # Arrange
    _xx, filt_xx, pha, _amp = _run_pipeline()
    # Act
    result = pha.shape
    # Assert
    assert result == filt_xx.shape
