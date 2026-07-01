#!/usr/bin/env python3
"""Tests for scitex_dsp._features.pac_features + the feature registry (P1a).

Covers:
- canonical 19-name order + names/values alignment with the registry
- 1:1 reuse of scitex_stats.descriptive for the 11 nan stats (composed
  naniqr/nanrange) and the 4 circular stats
- PROVISIONAL GMM closed-form math against hand-computed values, plus sane
  behaviour on a bimodal sample (sklearn importorskip'd)
- registry completeness + provisional flag semantics
- phase=None and include_gmm=False shortening behaviour
"""

from __future__ import annotations

import numpy as np
import pytest

from scitex_dsp import PAC_FEATURE_REGISTRY, feature_registry, pac_features

CANONICAL_ORDER = [
    "gmm_ashmans_d",
    "gmm_bhattacharyya_coeff",
    "gmm_bimodality_coeff",
    "gmm_weight_ratio",
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
    "phase_circular_concentration",
    "phase_circular_kurtosis",
    "phase_circular_mean",
    "phase_circular_skewness",
]

GMM_NAMES = CANONICAL_ORDER[:4]
NAN_NAMES = CANONICAL_ORDER[4:15]
CIRCULAR_NAMES = CANONICAL_ORDER[15:]


@pytest.fixture
def pac_z():
    rng = np.random.default_rng(0)
    return rng.normal(2.0, 1.5, size=200)


@pytest.fixture
def phase():
    rng = np.random.default_rng(1)
    return rng.uniform(0, 2 * np.pi, size=200)


# ---------------------------------------------------------------------------
# names / order / values alignment
# ---------------------------------------------------------------------------
def test_full_19_names_in_canonical_order(pac_z, phase):
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    out = pac_features(pac_z, phase, include_gmm=True)
    assert out["names"] == CANONICAL_ORDER
    assert len(out["names"]) == 19
    assert out["values"].shape == (19,)
    assert np.all(np.isfinite(out["values"]))


def test_names_values_align_with_registry(pac_z, phase):
    pytest.importorskip("sklearn")
    out = pac_features(pac_z, phase, include_gmm=True)
    for name in out["names"]:
        assert name in PAC_FEATURE_REGISTRY


def test_phase_none_skips_circular(pac_z):
    pytest.importorskip("sklearn")
    out = pac_features(pac_z, None, include_gmm=True)
    assert out["names"] == CANONICAL_ORDER[:15]
    assert out["values"].shape == (15,)
    assert not any(n.startswith("phase_circular") for n in out["names"])


def test_include_gmm_false_skips_gmm(pac_z, phase):
    out = pac_features(pac_z, phase, include_gmm=False)
    assert out["names"] == CANONICAL_ORDER[4:]
    assert out["values"].shape == (15,)
    assert not any(n.startswith("gmm_") for n in out["names"])


def test_no_gmm_no_phase_is_11(pac_z):
    out = pac_features(pac_z, None, include_gmm=False)
    assert out["names"] == CANONICAL_ORDER[4:15]
    assert len(out["names"]) == 11


# ---------------------------------------------------------------------------
# 1:1 reuse of scitex_stats.descriptive — nan family
# ---------------------------------------------------------------------------
def test_nan_stats_match_scitex_stats(pac_z):
    from scitex_stats.descriptive import (
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

    out = pac_features(pac_z, None, include_gmm=False)
    vals = dict(zip(out["names"], out["values"]))

    q25 = float(nanq25(pac_z, axis=-1))
    q75 = float(nanq75(pac_z, axis=-1))
    mx = float(nanmax(pac_z, axis=-1))
    mn = float(nanmin(pac_z, axis=-1))

    assert vals["nanmean"] == pytest.approx(float(nanmean(pac_z, axis=-1)))
    assert vals["nanstd"] == pytest.approx(float(nanstd(pac_z, axis=-1)))
    assert vals["nanmax"] == pytest.approx(mx)
    assert vals["nanmin"] == pytest.approx(mn)
    assert vals["nanq25"] == pytest.approx(q25)
    assert vals["nanq50"] == pytest.approx(float(nanq50(pac_z, axis=-1)))
    assert vals["nanq75"] == pytest.approx(q75)
    assert vals["nankurtosis"] == pytest.approx(float(nankurtosis(pac_z, axis=-1)))
    assert vals["nanskewness"] == pytest.approx(float(nanskewness(pac_z, axis=-1)))
    # composed
    assert vals["naniqr"] == pytest.approx(q75 - q25)
    assert vals["nanrange"] == pytest.approx(mx - mn)


# ---------------------------------------------------------------------------
# 1:1 reuse — circular family
# ---------------------------------------------------------------------------
def test_circular_stats_match_scitex_stats(phase):
    from scitex_stats.descriptive import (
        circular_concentration,
        circular_kurtosis,
        circular_mean,
        circular_skewness,
    )

    out = pac_features([1.0, 2.0, 3.0, 4.0], phase, include_gmm=False)
    vals = dict(zip(out["names"], out["values"]))

    angles2d = np.asarray(phase, dtype=float).reshape(1, -1)
    weights2d = np.ones_like(angles2d)

    # circular_* reduce a single batch row to a length-1 array; extract the
    # scalar the same way pac_features does internally so this stays a genuine
    # 1:1 equivalence check against the scitex_stats functions.
    def _scalar(x):
        return float(np.asarray(x).reshape(-1)[0])

    assert vals["phase_circular_mean"] == pytest.approx(
        _scalar(circular_mean(angles2d, weights2d, dim=-1))
    )
    assert vals["phase_circular_concentration"] == pytest.approx(
        _scalar(circular_concentration(angles2d, weights2d, dim=-1))
    )
    assert vals["phase_circular_skewness"] == pytest.approx(
        _scalar(circular_skewness(angles2d, weights2d, dim=-1))
    )
    assert vals["phase_circular_kurtosis"] == pytest.approx(
        _scalar(circular_kurtosis(angles2d, weights2d, dim=-1))
    )


# ---------------------------------------------------------------------------
# GMM family — PROVISIONAL closed forms
# ---------------------------------------------------------------------------
def test_gmm_closed_forms_hand_computed():
    """Feed a cleanly-separated bimodal sample; check the fitted-mode closed
    forms against the same formulas evaluated on the fitted (mu, sigma, w).

    We do NOT claim bit-exact agreement with neurovista's pipeline; we assert
    the module's closed-form math is internally consistent with the standard
    definitions applied to the sklearn fit.
    """
    pytest.importorskip("sklearn")
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(42)
    a = rng.normal(-5.0, 0.5, size=500)
    b = rng.normal(5.0, 1.0, size=500)
    x = np.concatenate([a, b])

    out = pac_features(x, None, include_gmm=True)
    vals = dict(zip(out["names"], out["values"]))

    # Re-fit identically (deterministic random_state=0) and hand-compute.
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(x.reshape(-1, 1))
    mu = gmm.means_.reshape(-1)
    sigma = np.sqrt(gmm.covariances_.reshape(-1))
    w = gmm.weights_.reshape(-1)
    mu0, mu1 = float(mu[0]), float(mu[1])
    s0, s1 = float(sigma[0]), float(sigma[1])
    w0, w1 = float(w[0]), float(w[1])
    var_sum = s0**2 + s1**2

    expected_ashmans = np.sqrt(2.0) * abs(mu0 - mu1) / np.sqrt(var_sum)
    expected_bhatt = np.sqrt(2.0 * s0 * s1 / var_sum) * np.exp(
        -((mu0 - mu1) ** 2) / (4.0 * var_sum)
    )
    expected_wr = min(w0, w1) / max(w0, w1)

    assert vals["gmm_ashmans_d"] == pytest.approx(expected_ashmans)
    assert vals["gmm_bhattacharyya_coeff"] == pytest.approx(expected_bhatt)
    assert vals["gmm_weight_ratio"] == pytest.approx(expected_wr)


def test_gmm_sane_on_bimodal():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    x = np.concatenate(
        [rng.normal(-4, 0.7, 400), rng.normal(4, 0.7, 400)]
    )
    out = pac_features(x, None, include_gmm=True)
    vals = dict(zip(out["names"], out["values"]))

    assert np.isfinite(vals["gmm_ashmans_d"])
    assert vals["gmm_ashmans_d"] > 0
    assert 0.0 < vals["gmm_weight_ratio"] <= 1.0
    # bhattacharyya coefficient is in [0, 1]
    assert 0.0 <= vals["gmm_bhattacharyya_coeff"] <= 1.0
    # clearly-bimodal Sarle coefficient should exceed the 5/9 uniform threshold
    assert vals["gmm_bimodality_coeff"] > 5.0 / 9.0


def test_bimodality_coeff_hand_computed():
    """Sarle's coefficient closed form on a known small sample."""
    from scitex_dsp._features._pac_features import _sample_bimodality_coeff

    x = np.array([1.0, 2.0, 2.0, 3.0, 8.0, 9.0, 9.0, 10.0])
    n = x.size
    z = (x - x.mean()) / x.std(ddof=0)
    m3 = np.mean(z**3)
    m4 = np.mean(z**4)
    g1, g2 = m3, m4 - 3.0
    G1 = g1 * np.sqrt(n * (n - 1)) / (n - 2)
    G2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6)
    expected = (G1**2 + 1.0) / (G2 + 3.0)

    assert _sample_bimodality_coeff(x) == pytest.approx(expected)


def test_gmm_missing_sklearn_raises(monkeypatch):
    """If sklearn is unavailable, include_gmm=True raises a clear ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="scikit-learn"):
        pac_features(np.arange(50.0), None, include_gmm=True)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_registry_covers_all_19_names():
    assert set(PAC_FEATURE_REGISTRY) == set(CANONICAL_ORDER)
    assert list(PAC_FEATURE_REGISTRY) == CANONICAL_ORDER


def test_registry_records_shape():
    expected_keys = {
        "family",
        "engine",
        "interpretation",
        "provisional",
        "input_provisional",
        "note",
    }
    for name, rec in PAC_FEATURE_REGISTRY.items():
        assert set(rec) == expected_keys
        assert isinstance(rec["interpretation"], str) and rec["interpretation"]
        assert isinstance(rec["provisional"], bool)
        assert isinstance(rec["input_provisional"], bool)
        assert isinstance(rec["note"], str)


def test_provisional_flag_true_exactly_for_gmm():
    """`provisional` (formula) true ONLY for the 4 GMM features."""
    provisional = {n for n, r in PAC_FEATURE_REGISTRY.items() if r["provisional"]}
    assert provisional == set(GMM_NAMES)


def test_input_provisional_flag_true_exactly_for_circular():
    """`input_provisional` true ONLY for the 4 circular features."""
    input_prov = {
        n for n, r in PAC_FEATURE_REGISTRY.items() if r["input_provisional"]
    }
    assert input_prov == set(CIRCULAR_NAMES)


def test_nan_family_both_flags_false():
    """The 11 nan features are fully locked: both flags False."""
    for name in NAN_NAMES:
        rec = PAC_FEATURE_REGISTRY[name]
        assert rec["provisional"] is False
        assert rec["input_provisional"] is False


def test_gmm_and_circular_notes_present():
    for name in GMM_NAMES:
        assert PAC_FEATURE_REGISTRY[name]["note"]
    for name in CIRCULAR_NAMES:
        assert PAC_FEATURE_REGISTRY[name]["note"]


def test_feature_registry_accessor_returns_copy():
    reg = feature_registry()
    assert reg == PAC_FEATURE_REGISTRY
    reg["gmm_ashmans_d"]["family"] = "MUTATED"
    assert PAC_FEATURE_REGISTRY["gmm_ashmans_d"]["family"] == "pac_z_gmm"


# EOF
