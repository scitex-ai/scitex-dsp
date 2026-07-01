#!/usr/bin/env python3
"""Feature registry for the canonical PAC-summary set.

Maps each of the 19 feature names -> a record with ``family``, ``engine``,
a one-line ``interpretation``, and two independent provenance flags:

- ``provisional``       — is the FORMULA / computation itself provisional?
- ``input_provisional`` — is the fn locked but the INPUT ARRAY still unpinned?

This is what makes the flat ``pac_features`` vector self-documenting for
downstream DataFrames and figures. Per neurovista's final word:

- ``pac_z_nan`` (11): fully locked — ``provisional=False``,
  ``input_provisional=False`` (scitex-stats descriptive fns on the flattened
  surrogate-z comodulogram).
- ``phase_circular`` (4): the scitex-stats FUNCTION is locked
  (``provisional=False``) but the INPUT ARRAY is still provisional
  (``input_provisional=True``): likely a per-window preferred-phase /
  phase-bin distribution, not yet pinned upstream.
- ``pac_z_gmm`` (4): the FORMULA is provisional (``provisional=True``): the
  standard closed form is used, but the exact sklearn ``GaussianMixture``
  params (``n_init``/seed/``covariance_type``), the bimodality basis, and the
  weight-ratio definition are unrecoverable from live repos.

The registry order IS the canonical descriptor order (GMM, nan, circular;
alphabetical within family).
"""

from __future__ import annotations

__all__ = ["PAC_FEATURE_REGISTRY", "feature_registry"]


_GMM_NOTE = (
    "standard closed form; exact sklearn GaussianMixture params "
    "(n_init/seed/covariance_type) + bimodality basis + weight_ratio def "
    "unrecoverable from live repos"
)
_CIRCULAR_NOTE = (
    "scitex-stats fn locked; input array provisional (likely per-window "
    "preferred-phase / phase-bin distribution)"
)


# Each record: name -> {family, engine, interpretation, provisional,
#                       input_provisional, note}.
# `provisional`       -> is the FORMULA itself provisional?
# `input_provisional` -> is the fn locked but the INPUT ARRAY unpinned?
# `engine` names the concrete computation source so provenance is explicit.
PAC_FEATURE_REGISTRY: dict = {
    # -- pac_z_gmm (formula PROVISIONAL) -----------------------------------
    "gmm_ashmans_d": {
        "family": "pac_z_gmm",
        "engine": "sklearn.mixture.GaussianMixture(2) closed-form",
        "interpretation": "Ashman's D separation of the 2 fitted pac_z modes.",
        "provisional": True,
        "input_provisional": False,
        "note": _GMM_NOTE,
    },
    "gmm_bhattacharyya_coeff": {
        "family": "pac_z_gmm",
        "engine": "sklearn.mixture.GaussianMixture(2) closed-form",
        "interpretation": "Bhattacharyya overlap of the 2 fitted pac_z modes.",
        "provisional": True,
        "input_provisional": False,
        "note": _GMM_NOTE,
    },
    "gmm_bimodality_coeff": {
        "family": "pac_z_gmm",
        "engine": "Sarle's coefficient on flattened pac_z samples",
        "interpretation": "Sarle's bimodality coefficient of the pac_z sample.",
        "provisional": True,
        "input_provisional": False,
        "note": _GMM_NOTE,
    },
    "gmm_weight_ratio": {
        "family": "pac_z_gmm",
        "engine": "sklearn.mixture.GaussianMixture(2) closed-form",
        "interpretation": "min/max mixing-weight ratio of the 2 fitted modes.",
        "provisional": True,
        "input_provisional": False,
        "note": _GMM_NOTE,
    },
    # -- pac_z_nan (fully locked) ------------------------------------------
    "naniqr": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive nanq75 - nanq25",
        "interpretation": "Interquartile range of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nankurtosis": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive.nankurtosis",
        "interpretation": "Excess kurtosis of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nanmax": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive.nanmax",
        "interpretation": "Maximum of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nanmean": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive.nanmean",
        "interpretation": "Mean of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nanmin": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive.nanmin",
        "interpretation": "Minimum of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nanq25": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive.nanq25",
        "interpretation": "25th percentile of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nanq50": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive.nanq50",
        "interpretation": "Median (50th percentile) of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nanq75": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive.nanq75",
        "interpretation": "75th percentile of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nanrange": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive nanmax - nanmin",
        "interpretation": "Range (max - min) of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nanskewness": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive.nanskewness",
        "interpretation": "Skewness of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    "nanstd": {
        "family": "pac_z_nan",
        "engine": "scitex_stats.descriptive.nanstd",
        "interpretation": "Standard deviation of pac_z (NaN-aware).",
        "provisional": False,
        "input_provisional": False,
        "note": "",
    },
    # -- phase_circular (fn locked, INPUT ARRAY provisional) ----------------
    "phase_circular_concentration": {
        "family": "phase_circular",
        "engine": "scitex_stats.descriptive.circular_concentration",
        "interpretation": "Mean resultant length (concentration) of phase.",
        "provisional": False,
        "input_provisional": True,
        "note": _CIRCULAR_NOTE,
    },
    "phase_circular_kurtosis": {
        "family": "phase_circular",
        "engine": "scitex_stats.descriptive.circular_kurtosis",
        "interpretation": "Circular kurtosis of the phase angles.",
        "provisional": False,
        "input_provisional": True,
        "note": _CIRCULAR_NOTE,
    },
    "phase_circular_mean": {
        "family": "phase_circular",
        "engine": "scitex_stats.descriptive.circular_mean",
        "interpretation": "Circular mean angle of the phase (radians, [0, 2pi]).",
        "provisional": False,
        "input_provisional": True,
        "note": _CIRCULAR_NOTE,
    },
    "phase_circular_skewness": {
        "family": "phase_circular",
        "engine": "scitex_stats.descriptive.circular_skewness",
        "interpretation": "Circular skewness of the phase angles.",
        "provisional": False,
        "input_provisional": True,
        "note": _CIRCULAR_NOTE,
    },
}


def feature_registry() -> dict:
    """Return the PAC feature registry (name -> record).

    Returns a shallow copy so callers cannot mutate the module-level table.
    """
    return {name: dict(record) for name, record in PAC_FEATURE_REGISTRY.items()}


# EOF
