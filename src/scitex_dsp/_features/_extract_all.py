#!/usr/bin/env python3
"""Multi-backend feature facade — one flat named vector over many engines.

``extract_all`` is the single entry point a downstream consumer (neurovista's
iEEG pipeline) calls to obtain a ``catch22_all``-style ``{names, values}``
descriptor *regardless of which engine computed each feature*. It runs one or
more pluggable backends, concatenates their per-backend ``{names, values}`` in
a STABLE, documented order, and merges every backend's registry entries so that
each emitted name is self-documenting (``family``, ``engine``,
``interpretation``, and the two-flag provenance semantics from P1a:
``provisional`` / ``input_provisional``).

Backends (shipped)
------------------
- ``"pac"``     — delegates to :func:`scitex_dsp._features.pac_features`. It
  needs a precomputed ``pac_z`` (and optional ``phase``), which are supplied
  via kwargs (``extract_all(pac_z=..., phase=..., sets=["pac"])``). Computing
  ``pac_z`` from a raw signal is deliberately *not* done here — that requires
  unpinned PAC params and is future work; requesting ``"pac"`` without
  ``pac_z`` raises a clear error. Reuses ``pac_features``' registry entries.
- ``"catch22"`` — wraps :mod:`pycatch22` (``catch22_all`` on the 1-D time
  series ``x``). Lazy import; missing -> clear ImportError naming the
  ``catch22`` extra. Registers its 22 canonical names into the merged registry
  (``family="catch22"``, ``engine="pycatch22"``, ``provisional=False``).

Deferred backends
-----------------
- ``gpac`` — a future backend that would compute PAC from a raw signal via
  ``scitex_nn.PAC`` (gpac). Not implemented here: gpac is not installed and
  wrapping it is speculative. Adding it later is a one-liner (register a
  callable in ``_BACKENDS``); see the extension-point note below.

Stable ordering guarantee
-------------------------
Output order is deterministic: backends run in the order given by ``sets``
(default: all available backends in ``_DEFAULT_SETS`` order), and within each
backend its own canonical order is preserved. Names are globally unique — the
pac and catch22 families already use non-colliding, family-prefixed names
(``pac_z_nanmean`` etc. vs the ``DN_``/``CO_``/``SB_`` catch22 short names), so
no re-prefixing is needed; a collision across backends is treated as an error.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ._pac_features import pac_features
from ._registry import feature_registry as _pac_feature_registry

__all__ = [
    "extract_all",
    "extract_all_registry",
    "AVAILABLE_BACKENDS",
]


# ---------------------------------------------------------------------------
# catch22 canonical names + short interpretations.
#
# These are the 22 features pycatch22.catch22_all emits, in its canonical
# order. We hardcode the names/interpretations so the registry is populated
# *without* needing pycatch22 importable (registry queries must work even when
# the optional engine is absent); the actual VALUES still come from pycatch22
# and we assert name-agreement at call time.
# ---------------------------------------------------------------------------
_CATCH22_NAMES: list[str] = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]

_CATCH22_INTERP: dict[str, str] = {
    "DN_HistogramMode_5": "Mode of z-scored distribution (5-bin histogram).",
    "DN_HistogramMode_10": "Mode of z-scored distribution (10-bin histogram).",
    "CO_f1ecac": "First 1/e crossing of the autocorrelation function.",
    "CO_FirstMin_ac": "First minimum of the autocorrelation function.",
    "CO_HistogramAMI_even_2_5": "Automutual information, m=2, 5 bins.",
    "CO_trev_1_num": "Time-reversibility statistic (trev), lag 1.",
    "MD_hrv_classic_pnn40": "Proportion of successive diffs exceeding 40 (pNN40).",
    "SB_BinaryStats_mean_longstretch1": "Longest stretch above the mean (binary).",
    "SB_TransitionMatrix_3ac_sumdiagcov": "Transition-matrix diagonal covariance.",
    "PD_PeriodicityWang_th0_01": "Wang periodicity measure (threshold 0.01).",
    "CO_Embed2_Dist_tau_d_expfit_meandiff": "Exp-fit residual of embedding distances.",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi": "First min of Gaussian automutual info.",
    "FC_LocalSimple_mean1_tauresrat": "Change in correlation length after forecasting.",
    "DN_OutlierInclude_p_001_mdrmd": "Positive-outlier timing distribution.",
    "DN_OutlierInclude_n_001_mdrmd": "Negative-outlier timing distribution.",
    "SP_Summaries_welch_rect_area_5_1": "Power in the lowest 20% of frequencies.",
    "SB_BinaryStats_diff_longstretch0": "Longest stretch of decreasing values.",
    "SB_MotifThree_quantile_hh": "Entropy of 3-letter symbolic motifs.",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": "Rescaled-range fluctuation slope.",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": "Detrended fluctuation-analysis slope.",
    "SP_Summaries_welch_rect_centroid": "Centroid of the Welch power spectrum.",
    "FC_LocalSimple_mean3_stderr": "Mean error from a 3-point rolling forecast.",
}


def _catch22_registry() -> dict:
    """Registry entries for the 22 catch22 features (engine present or not)."""
    return {
        name: {
            "family": "catch22",
            "engine": "pycatch22",
            "interpretation": _CATCH22_INTERP[name],
            "provisional": False,
            "input_provisional": False,
            "note": "",
        }
        for name in _CATCH22_NAMES
    }


# ---------------------------------------------------------------------------
# Backend implementations. Each backend is a callable
# ``(x, fs, **kwargs) -> {"names": [...], "values": np.ndarray}`` that also
# knows how to describe itself via a paired ``*_registry`` provider.
# ---------------------------------------------------------------------------
def _backend_pac(x, fs, **kwargs) -> dict:
    """PAC backend — delegate to ``pac_features`` on a precomputed ``pac_z``.

    ``pac_z`` (and optional ``phase``) must be supplied via kwargs; this
    backend does NOT derive ``pac_z`` from the raw signal ``x`` (that needs
    unpinned PAC params and is future work — see the ``gpac`` extension point).
    """
    if "pac_z" not in kwargs or kwargs["pac_z"] is None:
        raise ValueError(
            "extract_all(sets=[...'pac'...]) requires a precomputed 'pac_z' "
            "array passed as a keyword (e.g. extract_all(pac_z=..., "
            "phase=..., sets=['pac'])). Computing pac_z from the raw signal "
            "'x' is future work (it needs unpinned PAC params); pass the "
            "precomputed comodulogram instead."
        )
    pac_z = kwargs["pac_z"]
    phase = kwargs.get("phase")
    include_gmm = kwargs.get("include_gmm", True)
    return pac_features(pac_z, phase, include_gmm=include_gmm)


def _backend_catch22(x, fs, **kwargs) -> dict:
    """catch22 backend — wrap ``pycatch22.catch22_all`` on the 1-D series ``x``."""
    if x is None:
        raise ValueError(
            "extract_all(sets=[...'catch22'...]) requires the time series 'x' "
            "(the first positional argument) — got None."
        )
    try:
        import pycatch22
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise ImportError(
            "The 'catch22' backend requires pycatch22. Install it via "
            "`pip install scitex-dsp[all]` (or `pip install pycatch22`)."
        ) from exc

    series = np.asarray(x, dtype=float).reshape(-1).tolist()
    out = pycatch22.catch22_all(series)
    names = list(out["names"])
    values = np.asarray(out["values"], dtype=float)
    return {"names": names, "values": values}


# name -> (compute callable, registry-provider callable)
_BACKENDS: dict[str, tuple[Callable, Callable]] = {
    "pac": (_backend_pac, _pac_feature_registry),
    "catch22": (_backend_catch22, _catch22_registry),
    # Extension point: add a future backend by registering a
    #   "name": (compute_fn, registry_provider_fn)
    # pair here, e.g. "gpac": (_backend_gpac, _gpac_registry) once
    # scitex_nn.PAC (gpac) is available and its params are pinned. DEFERRED.
}

#: Backend names available to :func:`extract_all`, in canonical order.
AVAILABLE_BACKENDS: tuple[str, ...] = tuple(_BACKENDS)

#: Default ``sets`` when the caller passes ``sets=None`` — all available
#: backends, in canonical order. Documented so downstream ordering is stable.
_DEFAULT_SETS: tuple[str, ...] = AVAILABLE_BACKENDS


def extract_all(
    x=None,
    fs: Optional[float] = None,
    sets: Optional[list] = None,
    **kwargs,
) -> dict:
    """Run one or more feature backends and emit a single flat named vector.

    Parameters
    ----------
    x : array-like, optional
        The 1-D time series. Required by the ``"catch22"`` backend. The
        ``"pac"`` backend ignores ``x`` and instead consumes ``pac_z``/``phase``
        from ``**kwargs`` (see below).
    fs : float, optional
        Sampling frequency. Currently unused by the shipped backends (catch22
        is sampling-rate-agnostic; pac consumes a precomputed ``pac_z``) but
        threaded through for future backends (e.g. a raw-signal PAC backend).
    sets : list[str], optional
        Backend names to run, e.g. ``["pac", "catch22"]``. Defaults to all
        available backends (``AVAILABLE_BACKENDS``: ``("pac", "catch22")``) in
        that canonical order. Unknown names raise ``ValueError``.
    **kwargs
        Backend-specific inputs. The ``"pac"`` backend accepts ``pac_z``
        (required), ``phase`` (optional), and ``include_gmm`` (default True).

    Returns
    -------
    dict
        ``{"names": list[str], "values": np.ndarray}`` — the concatenation of
        each backend's ``{names, values}`` in a STABLE order: backends in the
        order of ``sets``, and within each backend its own canonical order.
        ``values`` is a 1-D float array aligned to ``names``; ``names`` is
        globally unique.

    Raises
    ------
    ValueError
        If ``sets`` names an unknown backend, if two backends emit the same
        name (a collision), or if a required backend input is missing (e.g.
        ``"pac"`` without ``pac_z``).
    ImportError
        If a requested backend's optional engine is absent (e.g. ``"catch22"``
        without pycatch22 — install the ``catch22`` extra).

    Notes
    -----
    - The returned registry (queryable via :func:`extract_all_registry`) is
      merged from every run backend so each emitted name is documented.
    - ``gpac`` (raw-signal PAC via ``scitex_nn.PAC``) is a DEFERRED backend;
      see the module docstring and the ``_BACKENDS`` extension point.
    """
    if sets is None:
        sets = list(_DEFAULT_SETS)
    else:
        sets = list(sets)

    unknown = [name for name in sets if name not in _BACKENDS]
    if unknown:
        raise ValueError(
            f"extract_all: unknown backend(s) {unknown!r}. "
            f"Available backends: {list(AVAILABLE_BACKENDS)!r}."
        )

    names: list[str] = []
    values_parts: list[np.ndarray] = []
    seen: set[str] = set()

    for name in sets:
        compute_fn, _registry_fn = _BACKENDS[name]
        out = compute_fn(x, fs, **kwargs)
        b_names = list(out["names"])
        b_values = np.asarray(out["values"], dtype=float).reshape(-1)

        collisions = seen.intersection(b_names)
        if collisions:
            raise ValueError(
                f"extract_all: backend {name!r} emits name(s) "
                f"{sorted(collisions)!r} that collide with an earlier backend. "
                "Backend feature names must be globally unique."
            )
        seen.update(b_names)
        names.extend(b_names)
        values_parts.append(b_values)

    if values_parts:
        values = np.concatenate(values_parts)
    else:
        values = np.empty(0, dtype=float)

    return {"names": names, "values": values}


def extract_all_registry(sets: Optional[list] = None) -> dict:
    """Return the merged feature registry for the given backend ``sets``.

    Merges each backend's registry entries (name -> record) so every name that
    :func:`extract_all` can emit is documented (``family``, ``engine``,
    ``interpretation``, ``provisional``, ``input_provisional``, ``note``).

    Parameters
    ----------
    sets : list[str], optional
        Backends whose registries to merge. Defaults to all available backends.

    Returns
    -------
    dict
        ``name -> record`` for every feature the selected backends emit.
    """
    if sets is None:
        sets = list(_DEFAULT_SETS)
    else:
        sets = list(sets)

    unknown = [name for name in sets if name not in _BACKENDS]
    if unknown:
        raise ValueError(
            f"extract_all_registry: unknown backend(s) {unknown!r}. "
            f"Available backends: {list(AVAILABLE_BACKENDS)!r}."
        )

    merged: dict = {}
    for name in sets:
        _compute_fn, registry_fn = _BACKENDS[name]
        for feat_name, record in registry_fn().items():
            if feat_name in merged:
                raise ValueError(
                    f"extract_all_registry: name {feat_name!r} is documented by "
                    f"more than one backend — registry entries must be unique."
                )
            merged[feat_name] = dict(record)
    return merged


# EOF
