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

REGISTRY_KEYS = {
    "family",
    "engine",
    "interpretation",
    "provisional",
    "input_provisional",
    "note",
}


@pytest.fixture
def pac_z():
    rng = np.random.default_rng(0)
    return rng.normal(2.0, 1.5, size=200)


@pytest.fixture
def phase():
    rng = np.random.default_rng(1)
    return rng.uniform(0, 2 * np.pi, size=200)


def _expected_nan(pac_z):
    """The 11 nan stats computed directly via scitex_stats.descriptive."""
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

    q25 = float(nanq25(pac_z, axis=-1))
    q75 = float(nanq75(pac_z, axis=-1))
    mx = float(nanmax(pac_z, axis=-1))
    mn = float(nanmin(pac_z, axis=-1))
    return {
        "nanmean": float(nanmean(pac_z, axis=-1)),
        "nanstd": float(nanstd(pac_z, axis=-1)),
        "nanmax": mx,
        "nanmin": mn,
        "nanq25": q25,
        "nanq50": float(nanq50(pac_z, axis=-1)),
        "nanq75": q75,
        "nankurtosis": float(nankurtosis(pac_z, axis=-1)),
        "nanskewness": float(nanskewness(pac_z, axis=-1)),
        "naniqr": q75 - q25,  # composed
        "nanrange": mx - mn,  # composed
    }


def _scalar(x):
    """circular_* reduce a single batch row to a length-1 array; extract the
    scalar the same way pac_features does internally so the equivalence check
    stays a genuine 1:1 comparison against the scitex_stats functions."""
    return float(np.asarray(x).reshape(-1)[0])


def _expected_circular(phase):
    from scitex_stats.descriptive import (
        circular_concentration,
        circular_kurtosis,
        circular_mean,
        circular_skewness,
    )

    angles2d = np.asarray(phase, dtype=float).reshape(1, -1)
    weights2d = np.ones_like(angles2d)
    return {
        "phase_circular_mean": _scalar(circular_mean(angles2d, weights2d, dim=-1)),
        "phase_circular_concentration": _scalar(
            circular_concentration(angles2d, weights2d, dim=-1)
        ),
        "phase_circular_skewness": _scalar(
            circular_skewness(angles2d, weights2d, dim=-1)
        ),
        "phase_circular_kurtosis": _scalar(
            circular_kurtosis(angles2d, weights2d, dim=-1)
        ),
    }


@pytest.fixture
def bimodal_gmm_vals():
    """pac_features GMM outputs + hand-computed expected values on a cleanly
    separated bimodal sample (deterministic; sklearn required)."""
    pytest.importorskip("sklearn")
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(42)
    a = rng.normal(-5.0, 0.5, size=500)
    b = rng.normal(5.0, 1.0, size=500)
    x = np.concatenate([a, b])

    out = pac_features(x, None, include_gmm=True)
    vals = dict(zip(out["names"], out["values"]))

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(x.reshape(-1, 1))
    mu = gmm.means_.reshape(-1)
    sigma = np.sqrt(gmm.covariances_.reshape(-1))
    w = gmm.weights_.reshape(-1)
    mu0, mu1 = float(mu[0]), float(mu[1])
    s0, s1 = float(sigma[0]), float(sigma[1])
    w0, w1 = float(w[0]), float(w[1])
    var_sum = s0**2 + s1**2

    expected = {
        "gmm_ashmans_d": np.sqrt(2.0) * abs(mu0 - mu1) / np.sqrt(var_sum),
        "gmm_bhattacharyya_coeff": np.sqrt(2.0 * s0 * s1 / var_sum)
        * np.exp(-((mu0 - mu1) ** 2) / (4.0 * var_sum)),
        "gmm_weight_ratio": min(w0, w1) / max(w0, w1),
    }
    return vals, expected


@pytest.fixture
def sane_gmm_vals():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(-4, 0.7, 400), rng.normal(4, 0.7, 400)])
    out = pac_features(x, None, include_gmm=True)
    return dict(zip(out["names"], out["values"]))


# ---------------------------------------------------------------------------
# names / order / values alignment
# ---------------------------------------------------------------------------
def test_full_19_names_in_canonical_order(pac_z, phase):
    # Arrange
    pytest.importorskip("sklearn")
    # Act
    out = pac_features(pac_z, phase, include_gmm=True)
    # Assert
    assert out["names"] == CANONICAL_ORDER


def test_full_19_names_count(pac_z, phase):
    # Arrange
    pytest.importorskip("sklearn")
    # Act
    out = pac_features(pac_z, phase, include_gmm=True)
    # Assert
    assert len(out["names"]) == 19


def test_full_19_values_shape(pac_z, phase):
    # Arrange
    pytest.importorskip("sklearn")
    # Act
    out = pac_features(pac_z, phase, include_gmm=True)
    # Assert
    assert out["values"].shape == (19,)


def test_full_19_values_finite(pac_z, phase):
    # Arrange
    pytest.importorskip("sklearn")
    # Act
    out = pac_features(pac_z, phase, include_gmm=True)
    # Assert
    assert np.all(np.isfinite(out["values"]))


def test_names_all_in_registry(pac_z, phase):
    # Arrange
    pytest.importorskip("sklearn")
    out = pac_features(pac_z, phase, include_gmm=True)
    # Act
    unknown = [n for n in out["names"] if n not in PAC_FEATURE_REGISTRY]
    # Assert
    assert unknown == []


def test_phase_none_names_are_first_15(pac_z):
    # Arrange
    pytest.importorskip("sklearn")
    # Act
    out = pac_features(pac_z, None, include_gmm=True)
    # Assert
    assert out["names"] == CANONICAL_ORDER[:15]


def test_phase_none_values_shape(pac_z):
    # Arrange
    pytest.importorskip("sklearn")
    # Act
    out = pac_features(pac_z, None, include_gmm=True)
    # Assert
    assert out["values"].shape == (15,)


def test_phase_none_drops_circular(pac_z):
    # Arrange
    pytest.importorskip("sklearn")
    out = pac_features(pac_z, None, include_gmm=True)
    # Act
    circular = [n for n in out["names"] if n.startswith("phase_circular")]
    # Assert
    assert circular == []


def test_include_gmm_false_names(pac_z, phase):
    # Arrange
    # Act
    out = pac_features(pac_z, phase, include_gmm=False)
    # Assert
    assert out["names"] == CANONICAL_ORDER[4:]


def test_include_gmm_false_values_shape(pac_z, phase):
    # Arrange
    # Act
    out = pac_features(pac_z, phase, include_gmm=False)
    # Assert
    assert out["values"].shape == (15,)


def test_include_gmm_false_drops_gmm(pac_z, phase):
    # Arrange
    out = pac_features(pac_z, phase, include_gmm=False)
    # Act
    gmm = [n for n in out["names"] if n.startswith("gmm_")]
    # Assert
    assert gmm == []


def test_no_gmm_no_phase_names(pac_z):
    # Arrange
    # Act
    out = pac_features(pac_z, None, include_gmm=False)
    # Assert
    assert out["names"] == CANONICAL_ORDER[4:15]


def test_no_gmm_no_phase_count(pac_z):
    # Arrange
    # Act
    out = pac_features(pac_z, None, include_gmm=False)
    # Assert
    assert len(out["names"]) == 11


# ---------------------------------------------------------------------------
# 1:1 reuse of scitex_stats.descriptive — nan family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAN_NAMES)
def test_nan_stat_matches_scitex_stats(pac_z, name):
    # Arrange
    expected = _expected_nan(pac_z)
    out = pac_features(pac_z, None, include_gmm=False)
    vals = dict(zip(out["names"], out["values"]))
    # Act
    got = vals[name]
    # Assert
    assert got == pytest.approx(expected[name])


# ---------------------------------------------------------------------------
# 1:1 reuse — circular family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CIRCULAR_NAMES)
def test_circular_stat_matches_scitex_stats(phase, name):
    # Arrange
    expected = _expected_circular(phase)
    out = pac_features([1.0, 2.0, 3.0, 4.0], phase, include_gmm=False)
    vals = dict(zip(out["names"], out["values"]))
    # Act
    got = vals[name]
    # Assert
    assert got == pytest.approx(expected[name])


# ---------------------------------------------------------------------------
# GMM family — PROVISIONAL closed forms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["gmm_ashmans_d", "gmm_bhattacharyya_coeff", "gmm_weight_ratio"]
)
def test_gmm_closed_form_matches_hand_computed(bimodal_gmm_vals, name):
    # Arrange
    vals, expected = bimodal_gmm_vals
    # Act
    got = vals[name]
    # Assert
    assert got == pytest.approx(expected[name])


def test_gmm_ashmans_d_finite(sane_gmm_vals):
    # Arrange
    # Act
    got = sane_gmm_vals["gmm_ashmans_d"]
    # Assert
    assert np.isfinite(got)


def test_gmm_ashmans_d_positive(sane_gmm_vals):
    # Arrange
    # Act
    got = sane_gmm_vals["gmm_ashmans_d"]
    # Assert
    assert got > 0


def test_gmm_weight_ratio_in_unit_interval(sane_gmm_vals):
    # Arrange
    # Act
    got = sane_gmm_vals["gmm_weight_ratio"]
    # Assert
    assert 0.0 < got <= 1.0


def test_gmm_bhattacharyya_in_unit_interval(sane_gmm_vals):
    # Arrange
    # Act
    got = sane_gmm_vals["gmm_bhattacharyya_coeff"]
    # Assert
    assert 0.0 <= got <= 1.0


def test_gmm_bimodality_exceeds_uniform_threshold(sane_gmm_vals):
    # Arrange
    # Act
    got = sane_gmm_vals["gmm_bimodality_coeff"]
    # Assert
    assert got > 5.0 / 9.0


def test_bimodality_coeff_hand_computed():
    # Arrange
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
    # Act
    got = _sample_bimodality_coeff(x)
    # Assert
    assert got == pytest.approx(expected)


def test_gmm_missing_sklearn_raises(monkeypatch):
    # Arrange
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Act
    # Assert
    with pytest.raises(ImportError, match="scikit-learn"):
        pac_features(np.arange(50.0), None, include_gmm=True)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_registry_covers_all_19_names_as_set():
    # Arrange
    # Act
    # Assert
    assert set(PAC_FEATURE_REGISTRY) == set(CANONICAL_ORDER)


def test_registry_preserves_canonical_order():
    # Arrange
    # Act
    # Assert
    assert list(PAC_FEATURE_REGISTRY) == CANONICAL_ORDER


def test_registry_record_keys():
    # Arrange
    # Act
    bad = {n: set(r) for n, r in PAC_FEATURE_REGISTRY.items() if set(r) != REGISTRY_KEYS}
    # Assert
    assert bad == {}


def test_registry_interpretation_nonempty():
    # Arrange
    # Act
    bad = [
        n
        for n, r in PAC_FEATURE_REGISTRY.items()
        if not (isinstance(r["interpretation"], str) and r["interpretation"])
    ]
    # Assert
    assert bad == []


def test_registry_provisional_is_bool():
    # Arrange
    # Act
    bad = [n for n, r in PAC_FEATURE_REGISTRY.items() if not isinstance(r["provisional"], bool)]
    # Assert
    assert bad == []


def test_registry_input_provisional_is_bool():
    # Arrange
    # Act
    bad = [
        n
        for n, r in PAC_FEATURE_REGISTRY.items()
        if not isinstance(r["input_provisional"], bool)
    ]
    # Assert
    assert bad == []


def test_provisional_flag_true_exactly_for_gmm():
    # Arrange
    # Act
    provisional = {n for n, r in PAC_FEATURE_REGISTRY.items() if r["provisional"]}
    # Assert
    assert provisional == set(GMM_NAMES)


def test_input_provisional_flag_true_exactly_for_circular():
    # Arrange
    # Act
    input_prov = {
        n for n, r in PAC_FEATURE_REGISTRY.items() if r["input_provisional"]
    }
    # Assert
    assert input_prov == set(CIRCULAR_NAMES)


def test_nan_family_formula_locked():
    # Arrange
    # Act
    provisional = [n for n in NAN_NAMES if PAC_FEATURE_REGISTRY[n]["provisional"]]
    # Assert
    assert provisional == []


def test_nan_family_input_locked():
    # Arrange
    # Act
    input_prov = [n for n in NAN_NAMES if PAC_FEATURE_REGISTRY[n]["input_provisional"]]
    # Assert
    assert input_prov == []


def test_gmm_notes_present():
    # Arrange
    # Act
    missing = [n for n in GMM_NAMES if not PAC_FEATURE_REGISTRY[n]["note"]]
    # Assert
    assert missing == []


def test_circular_notes_present():
    # Arrange
    # Act
    missing = [n for n in CIRCULAR_NAMES if not PAC_FEATURE_REGISTRY[n]["note"]]
    # Assert
    assert missing == []


def test_feature_registry_equals_registry():
    # Arrange
    # Act
    reg = feature_registry()
    # Assert
    assert reg == PAC_FEATURE_REGISTRY


def test_feature_registry_returns_copy():
    # Arrange
    reg = feature_registry()
    # Act
    reg["gmm_ashmans_d"]["family"] = "MUTATED"
    # Assert
    assert PAC_FEATURE_REGISTRY["gmm_ashmans_d"]["family"] == "pac_z_gmm"


# EOF
