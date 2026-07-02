#!/usr/bin/env python3
"""Canonical PAC-summary feature set (`pac_features`) + feature registry.

Summarises a per-window/per-channel PAC comodulogram of surrogate-z-scored
PAC (``pac_z``, flattened over the phase x amplitude grid) into a flat, named
scalar-feature vector — a ``catch22_all``-style ``{names, values}`` contract
that makes the descriptor self-documenting for downstream DataFrames/figures.

Motivation
----------
The downstream consumer (neurovista, an iEEG seizure pipeline) summarises each
comodulogram into **19 scalar stats/channel** -> a 304-dim (16 channels x 19)
descriptor. This module ships that summary as a single canonical function so
the exact feature set, ordering, and provenance live in one place.

Feature families (19 features total)
------------------------------------
- ``pac_z_nan`` (11): NaN-aware descriptive stats over the flattened
  ``pac_z`` samples. These map **1:1** onto ``scitex_stats.descriptive``
  (``naniqr``/``nanrange`` are composed from quantiles/extrema).
- ``phase_circular`` (4): circular descriptive stats over a caller-supplied
  ``phase`` angle array.
- ``pac_z_gmm`` (4): **PROVISIONAL** 2-component Gaussian-mixture summaries
  of the flattened ``pac_z`` distribution. The exact upstream pipeline code
  is not yet available; the closed forms here are the standard textbook
  definitions and are subject to pinning once neurovista confirms.

Canonical descriptor order (per-channel, alphabetical within family, GMM
first to match neurovista's 304-dim assembly)::

    [gmm_ashmans_d, gmm_bhattacharyya_coeff, gmm_bimodality_coeff,
     gmm_weight_ratio, naniqr, nankurtosis, nanmax, nanmean, nanmin,
     nanq25, nanq50, nanq75, nanrange, nanskewness, nanstd,
     phase_circular_concentration, phase_circular_kurtosis,
     phase_circular_mean, phase_circular_skewness]

Notes on inputs
---------------
- ``pac_z`` is the flattened comodulogram samples (1-D array). A trailing-axis
  batch is not assumed; pass one channel/window at a time.
- ``phase`` is a caller-supplied phase-angle array (radians) that feeds the
  circular stats. Its exact upstream source is still being confirmed by
  neurovista, so it is accepted as-is and documented rather than derived here.
  When ``phase is None`` the 4 circular features are **skipped** (and the
  returned ``names`` shortened accordingly) — the least-surprising behaviour
  for callers that only have ``pac_z``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from scitex_stats.descriptive import (
    circular_concentration,
    circular_kurtosis,
    circular_mean,
    circular_skewness,
    nankurtosis,
    nanmax,
    nanmean,
    nanmin,
    nanq25,
    nanq50,
    nanq75,
    nanskewness,
    nanstd,
)

from ._registry import PAC_FEATURE_REGISTRY, feature_registry

__all__ = [
    "pac_features",
    "PAC_FEATURE_REGISTRY",
    "feature_registry",
]


# ---------------------------------------------------------------------------
# Canonical descriptor order (must match the registry ordering).
# ---------------------------------------------------------------------------
_GMM_NAMES = [
    "gmm_ashmans_d",
    "gmm_bhattacharyya_coeff",
    "gmm_bimodality_coeff",
    "gmm_weight_ratio",
]
_NAN_NAMES = [
    "naniqr",
    "nankurtosis",
    "nanmax",
    "nanmean",
    "nanmin",
    "nanq25",
    "nanq50",
    "nanq75",
    "nanrange",
    "nanskewness",
    "nanstd",
]
_CIRCULAR_NAMES = [
    "phase_circular_concentration",
    "phase_circular_kurtosis",
    "phase_circular_mean",
    "phase_circular_skewness",
]


# ---------------------------------------------------------------------------
# nan-family engine: 1:1 reuse of scitex_stats.descriptive (naniqr/nanrange
# composed from quantiles/extrema per the scitex-stats surface).
# ---------------------------------------------------------------------------
def _nan_features(pac_z: np.ndarray) -> dict:
    """Compute the 11 NaN-aware descriptive stats over flattened ``pac_z``.

    ``naniqr`` and ``nanrange`` are composed (scitex-stats does not ship them
    directly): ``naniqr = nanq75 - nanq25`` and ``nanrange = nanmax - nanmin``.
    """
    q25 = float(nanq25(pac_z, axis=-1))
    q75 = float(nanq75(pac_z, axis=-1))
    mx = float(nanmax(pac_z, axis=-1))
    mn = float(nanmin(pac_z, axis=-1))
    return {
        "naniqr": q75 - q25,
        "nankurtosis": float(nankurtosis(pac_z, axis=-1)),
        "nanmax": mx,
        "nanmean": float(nanmean(pac_z, axis=-1)),
        "nanmin": mn,
        "nanq25": q25,
        "nanq50": float(nanq50(pac_z, axis=-1)),
        "nanq75": q75,
        "nanrange": mx - mn,
        "nanskewness": float(nanskewness(pac_z, axis=-1)),
        "nanstd": float(nanstd(pac_z, axis=-1)),
    }


# ---------------------------------------------------------------------------
# circular-family engine: 1:1 reuse of scitex_stats.descriptive circular stats.
# The scitex-stats circular funcs are histogram-weighted and require >=2-D
# input; we treat the supplied phase array as an unweighted sample of angles
# (uniform weights) and add a leading batch axis to satisfy the API, then
# squeeze the scalar result back out.
# ---------------------------------------------------------------------------
def _circular_features(phase: np.ndarray) -> dict:
    """Compute the 4 circular descriptive stats over a phase-angle array."""
    angles = np.asarray(phase, dtype=float).reshape(-1)
    # scitex-stats circular funcs require >=2-D (batch-first); use uniform
    # weights so the result is an unweighted circular statistic of the angles.
    # With a single batch row, the non-keepdims reduction over dim=-1 leaves a
    # length-1 batch axis, so flatten the result to a Python scalar.
    angles2d = angles[np.newaxis, :]
    weights2d = np.ones_like(angles2d)

    def _scalar(x):
        return float(np.asarray(x).reshape(-1)[0])

    return {
        "phase_circular_concentration": _scalar(
            circular_concentration(angles2d, weights2d, dim=-1)
        ),
        "phase_circular_kurtosis": _scalar(
            circular_kurtosis(angles2d, weights2d, dim=-1)
        ),
        "phase_circular_mean": _scalar(circular_mean(angles2d, weights2d, dim=-1)),
        "phase_circular_skewness": _scalar(
            circular_skewness(angles2d, weights2d, dim=-1)
        ),
    }


# ---------------------------------------------------------------------------
# GMM-family engine — PROVISIONAL.
#
# Standard closed forms on a 2-component 1-D Gaussian mixture fitted to the
# flattened pac_z samples. sklearn is an OPTIONAL/lazy dependency; if it is
# missing we raise a clear ImportError naming the missing dep rather than
# silently returning NaN, so callers know the GMM block was not computed.
# Every GMM quantity is PROVISIONAL — subject to neurovista pinning the exact
# pipeline code.
# ---------------------------------------------------------------------------
def _sample_bimodality_coeff(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient ``(skew^2 + 1) / kurt`` (sample-corrected).

    Computed directly on the flattened samples ``x`` (not the fitted mixture),
    using the sample-corrected skewness/kurtosis so the classic finite-sample
    denominator applies. Returns NaN for ``n < 4`` (correction undefined).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 4:
        return float("nan")
    mean = x.mean()
    std = x.std(ddof=0)
    if std == 0:
        return float("nan")
    z = (x - mean) / std
    m3 = np.mean(z**3)
    m4 = np.mean(z**4)
    # Sample-corrected skewness (g1 -> G1) and excess kurtosis (g2 -> G2).
    g1 = m3
    g2 = m4 - 3.0
    G1 = g1 * np.sqrt(n * (n - 1)) / (n - 2)
    G2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6)
    # Sarle uses full (non-excess) kurtosis in the denominator: kurt = G2 + 3.
    denom = G2 + 3.0
    if denom == 0:
        return float("nan")
    return float((G1**2 + 1.0) / denom)


def _gmm_features(pac_z: np.ndarray) -> dict:
    """Fit a 2-component 1-D GMM to flattened ``pac_z`` and summarise it.

    PROVISIONAL. Closed forms (components ``(mu0, sigma0, w0)``,
    ``(mu1, sigma1, w1)``):

    - ``gmm_ashmans_d = sqrt(2) * |mu0 - mu1| / sqrt(sigma0^2 + sigma1^2)``
    - ``gmm_bhattacharyya_coeff = sqrt(2 * sigma0 * sigma1 / (sigma0^2 + sigma1^2))
      * exp(-(mu0 - mu1)^2 / (4 * (sigma0^2 + sigma1^2)))``
    - ``gmm_bimodality_coeff`` = Sarle's coefficient on the raw samples.
    - ``gmm_weight_ratio = min(w0, w1) / max(w0, w1)``

    Raises
    ------
    ImportError
        If scikit-learn is not installed (install the ``[all]`` extra).
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise ImportError(
            "pac_features(..., include_gmm=True) requires scikit-learn for the "
            "PROVISIONAL GMM feature family. Install it via "
            "`pip install scitex-dsp[all]` (or `pip install scikit-learn`), "
            "or call pac_features(..., include_gmm=False) to skip the GMM block."
        ) from exc

    x = np.asarray(pac_z, dtype=float).reshape(-1)
    x = x[~np.isnan(x)]
    samples = x.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(samples)

    mu = gmm.means_.reshape(-1)
    var = gmm.covariances_.reshape(-1)
    sigma = np.sqrt(var)
    w = gmm.weights_.reshape(-1)

    mu0, mu1 = float(mu[0]), float(mu[1])
    s0, s1 = float(sigma[0]), float(sigma[1])
    w0, w1 = float(w[0]), float(w[1])

    var_sum = s0**2 + s1**2
    ashmans_d = np.sqrt(2.0) * abs(mu0 - mu1) / np.sqrt(var_sum)
    bhattacharyya = np.sqrt(2.0 * s0 * s1 / var_sum) * np.exp(
        -((mu0 - mu1) ** 2) / (4.0 * var_sum)
    )
    weight_ratio = min(w0, w1) / max(w0, w1)

    return {
        "gmm_ashmans_d": float(ashmans_d),
        "gmm_bhattacharyya_coeff": float(bhattacharyya),
        "gmm_bimodality_coeff": _sample_bimodality_coeff(x),
        "gmm_weight_ratio": float(weight_ratio),
    }


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def pac_features(
    pac_z,
    phase: Optional["np.ndarray"] = None,
    *,
    include_gmm: bool = True,
) -> dict:
    """Summarise a flattened PAC comodulogram into a named scalar-feature vector.

    Parameters
    ----------
    pac_z : array-like, shape (n,)
        The flattened ``n_pha x n_amp`` surrogate-z-scored PAC comodulogram
        for a single channel and single window, flattened over the
        phase x amplitude grid. 1-D input is expected; a trailing-axis batch
        is not assumed.
    phase : array-like, optional
        Caller-supplied phase-angle array (radians) feeding the 4 circular
        features. **PROVISIONAL input**: its exact upstream source is still
        being confirmed by neurovista (likely a per-window preferred-phase /
        phase-bin distribution), so it is accepted as-is and NOT derived here.
        If ``None`` (default), the 4 ``phase_circular_*`` features are
        **skipped** and ``names`` is shortened accordingly.
    include_gmm : bool, default True
        If True, compute the 4 PROVISIONAL ``pac_z_gmm`` features (requires
        scikit-learn). If False, they are skipped and ``names`` shortened.

    Returns
    -------
    dict
        ``{"names": list[str], "values": np.ndarray}`` — a flat, named,
        catch22-style descriptor. ``values`` is ordered to match ``names``,
        which follow the canonical descriptor order (GMM, nan, circular),
        with any skipped families omitted.

    Notes
    -----
    - The 11 ``pac_z_nan`` and 4 ``phase_circular`` features reuse
      ``scitex_stats.descriptive`` 1:1 (``naniqr``/``nanrange`` composed).
    - The 4 ``pac_z_gmm`` features are PROVISIONAL (see module docstring and
      the feature registry ``provisional`` flag).
    """
    pac_z_arr = np.asarray(pac_z, dtype=float).reshape(-1)

    values_by_name: dict = {}

    if include_gmm:
        values_by_name.update(_gmm_features(pac_z_arr))

    values_by_name.update(_nan_features(pac_z_arr))

    if phase is not None:
        values_by_name.update(_circular_features(phase))

    # Assemble in canonical order, omitting skipped families.
    names = []
    if include_gmm:
        names += _GMM_NAMES
    names += _NAN_NAMES
    if phase is not None:
        names += _CIRCULAR_NAMES

    values = np.array([values_by_name[n] for n in names], dtype=float)
    return {"names": names, "values": values}


# EOF
