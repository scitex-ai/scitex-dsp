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
def test_catch22_matches_direct_pycatch22(x):
    pycatch22 = pytest.importorskip("pycatch22")

    out = extract_all(x, sets=["catch22"])
    direct = pycatch22.catch22_all(np.asarray(x, dtype=float).reshape(-1).tolist())

    assert out["names"] == list(direct["names"])
    assert len(out["names"]) == 22
    assert out["values"].shape == (22,)
    np.testing.assert_allclose(
        out["values"], np.asarray(direct["values"], dtype=float)
    )


def test_catch22_every_name_has_registry_entry(x):
    pytest.importorskip("pycatch22")
    out = extract_all(x, sets=["catch22"])
    reg = extract_all_registry(sets=["catch22"])
    for name in out["names"]:
        assert name in reg
        assert reg[name]["family"] == "catch22"
        assert reg[name]["engine"] == "pycatch22"
        assert reg[name]["provisional"] is False


# ---------------------------------------------------------------------------
# pac backend == pac_features delegate parity
# ---------------------------------------------------------------------------
def test_pac_backend_equals_pac_features(pac_z, phase):
    pytest.importorskip("sklearn")
    out = extract_all(pac_z=pac_z, phase=phase, sets=["pac"])
    ref = pac_features(pac_z, phase)

    assert out["names"] == ref["names"]
    np.testing.assert_allclose(out["values"], ref["values"])


def test_pac_backend_include_gmm_false(pac_z, phase):
    out = extract_all(pac_z=pac_z, phase=phase, sets=["pac"], include_gmm=False)
    ref = pac_features(pac_z, phase, include_gmm=False)
    assert out["names"] == ref["names"]
    np.testing.assert_allclose(out["values"], ref["values"])


# ---------------------------------------------------------------------------
# concatenation of both backends
# ---------------------------------------------------------------------------
def test_concat_pac_and_catch22_stable_order(x, pac_z, phase):
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")

    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["pac", "catch22"])
    pac_ref = pac_features(pac_z, phase)
    cat_ref = extract_all(x, sets=["catch22"])

    # Stable order: pac's canonical order first, then catch22's canonical order.
    assert out["names"] == pac_ref["names"] + cat_ref["names"]
    # Unique names.
    assert len(out["names"]) == len(set(out["names"]))
    # Length = sum of parts.
    assert len(out["names"]) == len(pac_ref["names"]) + len(cat_ref["names"])
    assert out["values"].shape == (len(out["names"]),)
    # Values are the concatenation.
    np.testing.assert_allclose(
        out["values"],
        np.concatenate([pac_ref["values"], cat_ref["values"]]),
    )


def test_concat_reverse_order_is_reversed(x, pac_z, phase):
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["catch22", "pac"])
    pac_ref = pac_features(pac_z, phase)
    cat_ref = extract_all(x, sets=["catch22"])
    assert out["names"] == cat_ref["names"] + pac_ref["names"]


def test_concat_every_name_has_registry_entry(x, pac_z, phase):
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    out = extract_all(x, pac_z=pac_z, phase=phase, sets=["pac", "catch22"])
    reg = extract_all_registry(sets=["pac", "catch22"])
    for name in out["names"]:
        assert name in reg
        rec = reg[name]
        assert set(rec) >= {
            "family",
            "engine",
            "interpretation",
            "provisional",
            "input_provisional",
        }


def test_names_globally_unique_across_backends():
    """pac and catch22 families must not share any feature name."""
    reg = extract_all_registry(sets=["pac", "catch22"])
    # A collision would have raised in extract_all_registry; also sanity-check
    # the two name sets are disjoint.
    pac_reg = extract_all_registry(sets=["pac"])
    cat_reg = extract_all_registry(sets=["catch22"])
    assert set(pac_reg).isdisjoint(set(cat_reg))
    assert set(reg) == set(pac_reg) | set(cat_reg)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------
def test_pac_without_pac_z_raises(x):
    with pytest.raises(ValueError, match="pac_z"):
        extract_all(x, sets=["pac"])


def test_unknown_backend_raises(x):
    with pytest.raises(ValueError, match="unknown backend"):
        extract_all(x, sets=["not_a_backend"])


def test_unknown_backend_in_registry_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        extract_all_registry(sets=["nope"])


def test_catch22_missing_pycatch22_raises(monkeypatch, x):
    """If pycatch22 is unavailable, the catch22 backend raises a clear ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pycatch22" or name.startswith("pycatch22."):
            raise ImportError("No module named 'pycatch22'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="catch22"):
        extract_all(x, sets=["catch22"])


# ---------------------------------------------------------------------------
# defaults + introspection
# ---------------------------------------------------------------------------
def test_available_backends():
    assert AVAILABLE_BACKENDS == ("pac", "catch22")


def test_default_sets_runs_all_backends(x, pac_z, phase):
    """sets=None defaults to all available backends (pac, catch22) in order."""
    pytest.importorskip("pycatch22")
    pytest.importorskip("sklearn")
    out = extract_all(x, pac_z=pac_z, phase=phase)  # sets=None
    explicit = extract_all(x, pac_z=pac_z, phase=phase, sets=list(AVAILABLE_BACKENDS))
    assert out["names"] == explicit["names"]
    np.testing.assert_allclose(out["values"], explicit["values"])


def test_default_registry_covers_all_backends():
    reg = extract_all_registry()  # sets=None
    explicit = extract_all_registry(sets=list(AVAILABLE_BACKENDS))
    assert set(reg) == set(explicit)


def test_registry_is_a_copy():
    reg = extract_all_registry(sets=["catch22"])
    name = next(iter(reg))
    reg[name]["family"] = "MUTATED"
    fresh = extract_all_registry(sets=["catch22"])
    assert fresh[name]["family"] == "catch22"


# EOF
