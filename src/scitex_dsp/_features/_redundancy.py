#!/usr/bin/env python3
"""Feature-redundancy audit for the mixed PAC-19 + catch22-22 feature set.

Downstream (neurovista) summarises each iEEG window into the flat named vector
produced by :func:`scitex_dsp._features.extract_all` — a mix of the PAC-summary
families (``pac_z_nan`` / ``pac_z_gmm`` / ``phase_circular``) and the 22
``catch22`` features. Before feeding that descriptor to a classifier they want
to know how *redundant* it is: which features co-vary, whether within-family
correlation dominates cross-family correlation, and how much variance a few PCA
components capture. This module ships that audit as a small library so their
ad-hoc ``scripts/pac/stats/corr_pac19_catch22_*.py`` scripts can be retired.

Everything is driven by the feature REGISTRY family
(:func:`scitex_dsp._features.extract_all_registry`), so block grouping and
labels come from provenance metadata rather than hand-typed name lists. A
feature name that is absent from the registry is placed in an ``"other"`` block
rather than raising.

Capabilities
------------
- :func:`feature_correlation` — mixed-family correlation matrix (Spearman or
  Pearson) plus a family-block-ordered view and the block boundaries.
- :func:`redundancy_summary` — within-block vs cross-block mean/median ``|r|``.
- :func:`pca_loadings` — PCA loadings + explained-variance ratio on
  standardized features (with the variance-is-not-prediction caveat).
- :func:`correlation_by_group` — per-group correlation matrices and their
  difference, for epoch / latency-window comparisons (e.g. preictal vs ictal).

Dependencies
------------
- ``numpy`` / ``pandas`` / ``scipy`` are core dependencies.
- ``scipy.cluster.hierarchy`` provides the hierarchical-cluster ordering (core
  dep, always available).
- ``scikit-learn`` (the ``[features]`` extra) is required only by
  :func:`pca_loadings`; it is imported lazily and its absence raises a clear
  ``ImportError`` naming the extra.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "feature_correlation",
    "redundancy_summary",
    "pca_loadings",
    "correlation_by_group",
]


# ---------------------------------------------------------------------------
# Registry-driven family lookup.
# ---------------------------------------------------------------------------
_OTHER_BLOCK = "other"


def _family_of(names: Sequence[str]) -> list[str]:
    """Map each feature name to its registry ``family`` (``"other"`` if absent).

    The registry is queried once via
    :func:`scitex_dsp._features.extract_all_registry`; names not documented by
    any backend fall into the ``"other"`` block instead of raising.
    """
    from ._extract_all import extract_all_registry

    registry = extract_all_registry()
    families: list[str] = []
    for name in names:
        record = registry.get(name)
        family = record["family"] if record is not None else _OTHER_BLOCK
        families.append(family)
    return families


def _validate_X_names(X, names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Coerce ``X`` to a 2-D float array and validate it against ``names``."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"X must be 2-D (n_samples, n_features); got shape {arr.shape!r}."
        )
    names = list(names)
    if arr.shape[1] != len(names):
        raise ValueError(
            f"X has {arr.shape[1]} feature columns but {len(names)} names were "
            "given; they must match."
        )
    return arr, names


def _correlation_matrix(arr: np.ndarray, method: str) -> np.ndarray:
    """Return the (n_features, n_features) correlation matrix for ``method``."""
    method = method.lower()
    if method not in ("spearman", "pearson"):
        raise ValueError(
            f"method must be 'spearman' or 'pearson'; got {method!r}."
        )
    if method == "spearman":
        from scipy.stats import rankdata

        data = np.column_stack(
            [rankdata(arr[:, j]) for j in range(arr.shape[1])]
        )
    else:
        data = arr
    corr = np.corrcoef(data, rowvar=False)
    corr = np.atleast_2d(corr)
    # Constant columns yield NaN in corrcoef; treat a feature as uncorrelated
    # with others (0) but perfectly self-correlated (1) so the matrix stays
    # well-formed for downstream ordering/summaries.
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _block_order_indices(families: Sequence[str]) -> list[int]:
    """Order feature indices so identical families are contiguous.

    Families appear in first-seen order; indices within a family keep their
    original order. This is the "block" ordering that groups
    ``pac_z_nan`` / ``pac_z_gmm`` / ``phase_circular`` / ``catch22`` / ``other``
    into contiguous runs.
    """
    seen: list[str] = []
    for fam in families:
        if fam not in seen:
            seen.append(fam)
    order: list[int] = []
    for fam in seen:
        order.extend(i for i, f in enumerate(families) if f == fam)
    return order


def _cluster_order_indices(corr: np.ndarray) -> list[int]:
    """Hierarchical-cluster leaf order for a correlation matrix.

    Uses average linkage on a ``1 - |r|`` distance; returns the dendrogram leaf
    order so strongly-correlated features sit next to each other. Falls back to
    the identity order for degenerate (<=2 feature) matrices.
    """
    n = corr.shape[0]
    if n <= 2:
        return list(range(n))
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    return list(leaves_list(linkage_matrix))


def feature_correlation(
    X,
    names: Sequence[str],
    *,
    method: str = "spearman",
    block_order: bool = True,
) -> dict:
    """Mixed-family correlation matrix with registry block ordering.

    Parameters
    ----------
    X : array-like
        A 2-D array of shape ``(n_samples, n_features)`` (one row per window).
    names : sequence of str
        Feature names aligned to the columns of ``X`` (from ``extract_all``).
    method : str, optional
        ``"spearman"`` (default, rank correlation) or ``"pearson"``.
    block_order : bool, optional
        When ``True`` (default) the ordered view groups registry families into contiguous blocks and hierarchically clusters within each block; when ``False`` the ordered view equals the input order.

    Returns
    -------
    dict
        A mapping with keys ``"matrix"`` (the correlation ``DataFrame`` in input order), ``"ordered_matrix"`` (the block/cluster-ordered ``DataFrame``), ``"ordered_names"`` (names in the ordered layout), ``"families"`` (registry family per input name), ``"ordered_families"`` (family per ordered name), ``"block_boundaries"`` (dict family to ``(start, stop)`` index range in the ordered layout) and ``"method"``.

    Notes
    -----
    Block grouping and labels come from :func:`extract_all_registry`; names absent from the registry fall into the ``"other"`` block.
    """
    arr, names = _validate_X_names(X, names)
    families = _family_of(names)
    corr = _correlation_matrix(arr, method)
    matrix = pd.DataFrame(corr, index=names, columns=names)

    if block_order:
        block_idx = _block_order_indices(families)
        block_families = [families[i] for i in block_idx]
        seen: list[str] = []
        for fam in block_families:
            if fam not in seen:
                seen.append(fam)
        ordered_idx: list[int] = []
        for fam in seen:
            members = [
                block_idx[k] for k, f in enumerate(block_families) if f == fam
            ]
            sub = corr[np.ix_(members, members)]
            sub_order = _cluster_order_indices(sub)
            ordered_idx.extend(members[j] for j in sub_order)
    else:
        ordered_idx = list(range(len(names)))

    ordered_names = [names[i] for i in ordered_idx]
    ordered_families = [families[i] for i in ordered_idx]
    ordered_matrix = matrix.iloc[ordered_idx, ordered_idx]

    boundaries: dict = {}
    for pos, fam in enumerate(ordered_families):
        if fam not in boundaries:
            boundaries[fam] = [pos, pos + 1]
        else:
            boundaries[fam][1] = pos + 1
    block_boundaries = {fam: (lo, hi) for fam, (lo, hi) in boundaries.items()}

    return {
        "matrix": matrix,
        "ordered_matrix": ordered_matrix,
        "ordered_names": ordered_names,
        "families": families,
        "ordered_families": ordered_families,
        "block_boundaries": block_boundaries,
        "method": method.lower(),
    }


def redundancy_summary(
    X,
    names: Sequence[str],
    *,
    method: str = "spearman",
) -> pd.DataFrame:
    """Within-block vs cross-block ``|r|`` summary statistics.

    Computes, per registry family, the mean and median absolute correlation
    among that family's own features (within-block), plus a pooled within-block
    row and a pooled cross-block row (correlations between features of
    *different* families). This is the sanity table neurovista tracks (pooled
    within-PAC and within-catch22 typically exceed the cross-family level).

    Parameters
    ----------
    X : array-like
        A 2-D array of shape ``(n_samples, n_features)``.
    names : sequence of str
        Feature names aligned to the columns of ``X``.
    method : str, optional
        ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    pandas.DataFrame
        One row per family plus ``"within_all"`` and ``"cross"`` rows, with columns ``"scope"``, ``"family"``, ``"n_pairs"``, ``"mean_abs_r"`` and ``"median_abs_r"``. The mean/median are ``NaN`` when a scope has no eligible feature pairs.
    """
    arr, names = _validate_X_names(X, names)
    families = _family_of(names)
    corr = _correlation_matrix(arr, method)
    fam_arr = np.asarray(families)

    n = corr.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    abs_r = np.abs(corr[iu, ju])
    same_family = fam_arr[iu] == fam_arr[ju]

    rows: list[dict] = []

    def _stats(scope: str, family: str, mask: np.ndarray) -> dict:
        vals = abs_r[mask]
        return {
            "scope": scope,
            "family": family,
            "n_pairs": int(vals.size),
            "mean_abs_r": float(np.mean(vals)) if vals.size else float("nan"),
            "median_abs_r": float(np.median(vals)) if vals.size else float("nan"),
        }

    for fam in dict.fromkeys(families):
        mask = same_family & (fam_arr[iu] == fam)
        rows.append(_stats("within_family", fam, mask))

    rows.append(_stats("within_all", "*", same_family))
    rows.append(_stats("cross", "*", ~same_family))

    return pd.DataFrame(
        rows,
        columns=["scope", "family", "n_pairs", "mean_abs_r", "median_abs_r"],
    )


def pca_loadings(
    X,
    names: Sequence[str],
    *,
    n_components: Optional[int] = None,
    standardize: bool = True,
) -> dict:
    """PCA loadings and explained-variance ratio on the feature set.

    .. warning::

       Variance is not prediction. A component that captures a large share of
       the feature variance is not necessarily predictive of the outcome
       (e.g. seizure onset). PCA is an unsupervised redundancy/geometry probe;
       use it to see which features move together, not to rank features by
       predictive value. Pair it with a supervised model for that question.

    Parameters
    ----------
    X : array-like
        A 2-D array of shape ``(n_samples, n_features)``.
    names : sequence of str
        Feature names aligned to the columns of ``X``.
    n_components : int, optional
        Number of components; defaults to ``min(n_samples, n_features)``.
    standardize : bool, optional
        When ``True`` (default) each feature is z-scored (zero mean, unit variance) before PCA so high-variance-unit features do not dominate.

    Returns
    -------
    dict
        A mapping with keys ``"loadings"`` (a ``DataFrame`` of shape ``(n_components, n_features)`` indexed ``PC1..PCk``), ``"explained_variance_ratio"`` (a ``Series`` indexed ``PC1..PCk``), ``"n_components"``, ``"standardize"`` and ``"caveat"`` (the variance-is-not-prediction reminder as a string).

    Raises
    ------
    ImportError
        If scikit-learn is not installed; install the ``[features]`` extra via ``pip install scitex-dsp[features]``.
    """
    arr, names = _validate_X_names(X, names)
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise ImportError(
            "pca_loadings requires scikit-learn. Install the features extra: "
            "`pip install scitex-dsp[features]` (or `pip install scikit-learn`)."
        ) from exc

    data = arr
    if standardize:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std = np.where(std == 0.0, 1.0, std)
        data = (data - mean) / std

    max_components = min(data.shape)
    if n_components is None:
        n_components = max_components
    n_components = int(min(n_components, max_components))

    pca = PCA(n_components=n_components)
    pca.fit(data)

    component_labels = [f"PC{i + 1}" for i in range(n_components)]
    loadings = pd.DataFrame(
        pca.components_, index=component_labels, columns=names
    )
    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=component_labels,
        name="explained_variance_ratio",
    )
    caveat = (
        "Variance is not prediction: a high-variance PCA component is not "
        "necessarily predictive of the outcome. Use PCA as an unsupervised "
        "redundancy probe, not a feature-importance ranking."
    )

    return {
        "loadings": loadings,
        "explained_variance_ratio": explained,
        "n_components": n_components,
        "standardize": standardize,
        "caveat": caveat,
    }


def correlation_by_group(
    X,
    names: Sequence[str],
    groups,
    *,
    method: str = "spearman",
) -> dict:
    """Per-group correlation matrices and their difference.

    Splits the samples (rows) of ``X`` by ``groups`` and computes one
    correlation matrix per group, then a difference matrix. This lets callers
    compare epoch / latency windows — e.g. preictal (-16..-1 min) vs ictal
    (0..+2 min) — and inspect ``corr(preictal) - corr(ictal)``.

    Parameters
    ----------
    X : array-like
        A 2-D array of shape ``(n_samples, n_features)``.
    names : sequence of str
        Feature names aligned to the columns of ``X``.
    groups : array-like
        Per-sample group assignment of length ``n_samples``. A boolean mask is treated as two groups (``False``, ``True``); any hashable labels also work.
    method : str, optional
        ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    dict
        A mapping with keys ``"matrices"`` (dict group-label to correlation ``DataFrame``), ``"group_labels"`` (sorted list of group labels), ``"difference"`` (the first-minus-second group ``DataFrame`` when exactly two groups are present, else ``None``), ``"difference_pair"`` (the ``(first, second)`` label tuple used for the difference, else ``None``) and ``"method"``.

    Raises
    ------
    ValueError
        If ``groups`` length does not match the number of samples, or a group has fewer than two samples.
    """
    arr, names = _validate_X_names(X, names)
    groups_arr = np.asarray(groups)
    if groups_arr.ndim != 1 or groups_arr.shape[0] != arr.shape[0]:
        raise ValueError(
            f"groups must be 1-D of length n_samples ({arr.shape[0]}); got "
            f"shape {groups_arr.shape!r}."
        )
    if groups_arr.dtype == bool:
        group_labels = [g for g in (False, True) if np.any(groups_arr == g)]
    else:
        group_labels = sorted(np.unique(groups_arr).tolist())

    matrices: dict = {}
    for label in group_labels:
        mask = groups_arr == label
        if int(mask.sum()) < 2:
            raise ValueError(
                f"group {label!r} has fewer than 2 samples; cannot correlate."
            )
        sub = _correlation_matrix(arr[mask], method)
        matrices[label] = pd.DataFrame(sub, index=names, columns=names)

    difference = None
    difference_pair = None
    if len(group_labels) == 2:
        first, second = group_labels[0], group_labels[1]
        difference = matrices[first] - matrices[second]
        difference_pair = (first, second)

    return {
        "matrices": matrices,
        "group_labels": list(group_labels),
        "difference": difference,
        "difference_pair": difference_pair,
        "method": method.lower(),
    }


# EOF
