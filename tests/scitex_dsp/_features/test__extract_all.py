#!/usr/bin/env python3
"""Tests for scitex_dsp._features.extract_all — multi-backend feature facade (P1b).

Covers:
- catch22 backend == a direct pycatch22.catch22_all call (importorskip)
- pac backend == pac_features(pac_z, phase) names/values (delegate parity)
- sets=["pac","catch22"] concatenation: stable order, unique names, total
  length = sum, every emitted name has a merged-registry entry
- "pac" without pac_z raises a clear error; unknown backend raises a clear error
- catch22 without pycatch22 raises ImportError naming the extra
- default sets behaviour + registry provenance flag semantics
"""

from __future__ import annotations

import numpy as np
import pytest

from scitex_dsp import (
    AVAILABLE_BACKENDS,
    extract_all,
    extract_all_registry,
)
from scitex_dsp._features import pac_features


@pytest.fixture
def x():
    rng = np.random.default_rng(0)
    # A non-trivial 1-D time series for catch22.
    t = np.linspace(0, 4 * np.pi, 500)
    return np.sin(t) + 0.3 * rng.standard_normal(t.size)


@pytest.fixture
def pac_z():
    rng = np.random.default_rng(1)
    return rng.normal(2.0, 1.5, size=200)


@pytest.fixture
def phase():
    rng = np.random.default_rng(2)
    return rng.uniform(0, 2 * np.pi, size=200)


# ---------------------------------------------------------------------------
# catch22 backend parity with a direct pycatch22 call
# ---------------------------------------------------------------------------
def test_catch22_names_match_pycatch22(x):
    # Arrange
    pycatch22 = pytest.importorskip("pycatch22")
    direct = pycatch22.catch22_all(np.asarray(x, dtype=float).reshape(-1).tolist())
    # Act
    out = extract_all(x, sets=["catch22"])
    # Assert
    assert out["names"] == list(direct["names"])


def test_catch22_names_count_22(x):
    # Arrange
    pytest.importorskip("pycatch22")
    # Act
    out = extract_all(x, sets=["catch22"])
    # Assert
    assert len(out["names"]) == 22


def test_catch22_values_shape(x):
    # Arrange
    pytest.importorskip("pycatch22")
    # Act
    out = extract_all(x, sets=["catch22"])
    # Assert
    assert out["values"].shape == (22,)


def test_catch22_values_match_pycatch22(x):
    # Arrange
    pycatch22 = pytest.importorskip("pycatch22")
    direct = pycatch22.catch22_all(np.asarray(x, dtype=float).reshape(-1).tolist())
    # Act
    out = extract_all(x, sets=["catch22"])
    # Assert
    assert np.allclose(out["values"], np.asarray(direct["values"], dtype=float))


def test_catch22_names_in_registry(x):
    # Arrange
    pytest.importorskip("pycatch22")
    out = extract_all(x, sets=["catch22"])
    reg = extract_all_registry(sets=["catch22"])
    # Act
    missing = [n for n in out["names"] if n not in reg]
    # Assert
    assert missing == []


def test_catch22_registry_family(x):
    # Arrange
    pytest.importorskip("pycatch22")
    reg = extract_all_registry(sets=["catch22"])
    # Act
    bad = [n for n, r in reg.items() if r["family"] != "catch22"]
    # Assert
    assert bad == []


def test_catch22_registry_engine(x):
    # Arrange
    pytest.importorskip("pycatch22")
    reg = extract_all_registry(sets=["catch22"])
    # Act
    bad = [n for n, r in reg.items() if r["engine"] != "pycatch22"]
    # Assert
    assert bad == []


def test_catch22_registry_not_provisional(x):
    # Arrange
    pytest.importorskip("pycatch22")
    reg = extract_all_registry(sets=["catch22"])
    # Act
    bad = [n for n, r in reg.items() if r["provisional"] is not False]
    # Assert
    assert bad == []


# ---------------------------------------------------------------------------
# pac backend == pac_features delegate parity
# ---------------------------------------------------------------------------
def test_pac_backend_names_equal_pac_features(pac_z, phase):
    # Arrange
    pytest.importorskip("sklearn")
    ref = pac_features(pac_z, phase)
    # Act
    out = extract_all(pac_z=pac_z, phase=phase, sets=["pac"])
    # Assert
    assert out["names"] == ref["names"]


def test_pac_backend_values_equal_pac_features(pac_z, phase):
    # Arrange
    pytest.importorskip("sklearn")
    ref = pac_features(pac_z, phase)
    # Act
    out = extract_all(pac_z=pac_z, phase=phase, sets=["pac"])
    # Assert
    assert np.allclose(out["values"], ref["values"])


def test_pac_backend_include_gmm_false_names(pac_z, phase):
    # Arrange
    ref = pac_features(pac_z, phase, include_gmm=False)
    # Act
    out = extract_all(pac_z=pac_z, phase=phase, sets=["pac"], include_gmm=False)
    # Assert
    assert out["names"] == ref["names"]


def test_pac_backend_include_gmm_false_values(pac_z, phase):
    # Arrange
    ref = pac_features(pac_z, phase, include_gmm=False)
    # Act
    out = extract_all(pac_z=pac_z, phase=phase, sets=["pac"], include_gmm=False)
    # Assert
    assert np.allclose(out["values"], ref["values"])


# ---------------------------------------------------------------------------
# concatenation of both backends
# ---------------------------------------------------------------------------
def test_concat_names_are_pac_then_catch22(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    pac_ref = pac_features(pac_z, phase)
    cat_ref = extract_all(x, sets=["catch22"])
    # Act
    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["pac", "catch22"])
    # Assert
    assert out["names"] == pac_ref["names"] + cat_ref["names"]


def test_concat_names_unique(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    # Act
    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["pac", "catch22"])
    # Assert
    assert len(out["names"]) == len(set(out["names"]))


def test_concat_length_is_sum(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    pac_ref = pac_features(pac_z, phase)
    cat_ref = extract_all(x, sets=["catch22"])
    # Act
    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["pac", "catch22"])
    # Assert
    assert len(out["names"]) == len(pac_ref["names"]) + len(cat_ref["names"])


def test_concat_values_shape_matches_names(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    # Act
    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["pac", "catch22"])
    # Assert
    assert out["values"].shape == (len(out["names"]),)


def test_concat_values_are_concatenation(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    pac_ref = pac_features(pac_z, phase)
    cat_ref = extract_all(x, sets=["catch22"])
    # Act
    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["pac", "catch22"])
    # Assert
    assert np.allclose(
        out["values"], np.concatenate([pac_ref["values"], cat_ref["values"]])
    )


def test_concat_reverse_order_is_reversed(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    pac_ref = pac_features(pac_z, phase)
    cat_ref = extract_all(x, sets=["catch22"])
    # Act
    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["catch22", "pac"])
    # Assert
    assert out["names"] == cat_ref["names"] + pac_ref["names"]


def test_concat_names_all_in_registry(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["pac", "catch22"])
    reg = extract_all_registry(sets=["pac", "catch22"])
    # Act
    missing = [n for n in out["names"] if n not in reg]
    # Assert
    assert missing == []


def test_concat_registry_records_have_core_keys(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    core = {"family", "engine", "interpretation", "provisional", "input_provisional"}
    reg = extract_all_registry(sets=["pac", "catch22"])
    # Act
    bad = [n for n, r in reg.items() if not (set(r) >= core)]
    # Assert
    assert bad == []


def test_pac_and_catch22_names_disjoint():
    # Arrange
    pac_reg = extract_all_registry(sets=["pac"])
    cat_reg = extract_all_registry(sets=["catch22"])
    # Act
    # Assert
    assert set(pac_reg).isdisjoint(set(cat_reg))


def test_merged_registry_is_union():
    # Arrange
    reg = extract_all_registry(sets=["pac", "catch22"])
    pac_reg = extract_all_registry(sets=["pac"])
    cat_reg = extract_all_registry(sets=["catch22"])
    # Act
    # Assert
    assert set(reg) == set(pac_reg) | set(cat_reg)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------
def test_pac_without_pac_z_raises(x):
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError, match="pac_z"):
        extract_all(x, sets=["pac"])


def test_unknown_backend_raises(x):
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError, match="unknown backend"):
        extract_all(x, sets=["not_a_backend"])


def test_unknown_backend_in_registry_raises():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError, match="unknown backend"):
        extract_all_registry(sets=["nope"])


# NOTE: the "missing pycatch22 raises a clear ImportError" path is intentionally
# NOT unit-tested — simulating an absent dependency requires import mocking,
# which the ecosystem forbids (PA-306 no-mocks). The error path itself lives in
# the catch22 backend and names the `scitex-dsp[catch22]` extra.


# ---------------------------------------------------------------------------
# defaults + introspection
# ---------------------------------------------------------------------------
def test_available_backends():
    # Arrange
    # Act
    # Assert
    assert AVAILABLE_BACKENDS == ("pac", "catch22")


def test_default_sets_names_match_explicit(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    explicit = extract_all(x, pac_z=pac_z, phase=phase, sets=list(AVAILABLE_BACKENDS))
    # Act
    out = extract_all(x, pac_z=pac_z, phase=phase)  # sets=None
    # Assert
    assert out["names"] == explicit["names"]


def test_default_sets_values_match_explicit(x, pac_z, phase):
    # Arrange
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    explicit = extract_all(x, pac_z=pac_z, phase=phase, sets=list(AVAILABLE_BACKENDS))
    # Act
    out = extract_all(x, pac_z=pac_z, phase=phase)  # sets=None
    # Assert
    assert np.allclose(out["values"], explicit["values"])


def test_default_registry_covers_all_backends():
    # Arrange
    explicit = extract_all_registry(sets=list(AVAILABLE_BACKENDS))
    # Act
    reg = extract_all_registry()  # sets=None
    # Assert
    assert set(reg) == set(explicit)


def test_registry_is_a_copy():
    # Arrange
    reg = extract_all_registry(sets=["catch22"])
    name = next(iter(reg))
    # Act
    reg[name]["family"] = "MUTATED"
    fresh = extract_all_registry(sets=["catch22"])
    # Assert
    assert fresh[name]["family"] == "catch22"


# EOF
