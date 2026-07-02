#!/usr/bin/env python3
"""Tests for scitex_dsp._features._redundancy — feature-redundancy audit.

Covers the neurovista (a)-(d) spec:
- (a) mixed-family correlation matrix + registry block ordering
- (b) within-block vs cross-block |r| summary
- (c) PCA loadings + explained-variance ratio (variance != prediction)
- (d) per-group correlation matrices + difference matrix

Uses a synthetic block-correlated dataset whose column names are drawn from
the extract_all registry so families come from provenance, not hand-labels.
"""

from __future__ import annotations

import numpy as np
import pytest

from scitex_dsp import (
    correlation_by_group,
    feature_correlation,
    pca_loadings,
    redundancy_summary,
)


# ---------------------------------------------------------------------------
# Fixtures — a block-correlated synthetic dataset over real registry names.
# ---------------------------------------------------------------------------
@pytest.fixture
def pac_names():
    # Four pac_z_nan registry names (one family block).
    return ["nanmean", "nanstd", "nanmax", "nanmin"]


@pytest.fixture
def catch22_names():
    # Four catch22 registry names (a second family block).
    return [
        "DN_HistogramMode_5",
        "DN_HistogramMode_10",
        "CO_f1ecac",
        "CO_FirstMin_ac",
    ]


@pytest.fixture
def names(pac_names, catch22_names):
    return pac_names + catch22_names


@pytest.fixture
def block_X(names):
    # Two latent factors: one drives the pac block, one drives the catch22
    # block. Within-block correlation is high; cross-block is low.
    rng = np.random.default_rng(7)
    n_samples = 400
    factor_pac = rng.standard_normal(n_samples)
    factor_c22 = rng.standard_normal(n_samples)
    cols = []
    for j in range(4):  # pac block
        cols.append(factor_pac + 0.15 * rng.standard_normal(n_samples))
    for j in range(4):  # catch22 block
        cols.append(factor_c22 + 0.15 * rng.standard_normal(n_samples))
    return np.column_stack(cols)


@pytest.fixture
def corr_result(block_X, names):
    return feature_correlation(block_X, names, method="spearman")


# ---------------------------------------------------------------------------
# (a) correlation matrix — shape, symmetry, diagonal, ordering
# ---------------------------------------------------------------------------
def test_correlation_matrix_is_square(corr_result, names):
    # Arrange
    matrix = corr_result["matrix"]
    # Act
    shape = matrix.shape
    # Assert
    assert shape == (len(names), len(names))


def test_correlation_matrix_is_symmetric(corr_result):
    # Arrange
    matrix = corr_result["matrix"].to_numpy()
    # Act
    asymmetry = np.abs(matrix - matrix.T).max()
    # Assert
    assert asymmetry < 1e-9


def test_correlation_matrix_diagonal_is_one(corr_result):
    # Arrange
    matrix = corr_result["matrix"].to_numpy()
    # Act
    diagonal = np.diag(matrix)
    # Assert
    assert np.allclose(diagonal, 1.0)


def test_correlation_values_within_unit_range(corr_result):
    # Arrange
    matrix = corr_result["matrix"].to_numpy()
    # Act
    peak = np.abs(matrix).max()
    # Assert
    assert peak <= 1.0 + 1e-9


def test_pearson_method_is_accepted(block_X, names):
    # Arrange
    result = feature_correlation(block_X, names, method="pearson")
    # Act
    method = result["method"]
    # Assert
    assert method == "pearson"


def test_invalid_method_raises_value_error(block_X, names):
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        feature_correlation(block_X, names, method="kendall")


def test_mismatched_names_length_raises(block_X):
    # Arrange
    bad_names = ["only_one_name"]
    # Act
    # Assert
    with pytest.raises(ValueError):
        feature_correlation(block_X, bad_names)


# ---------------------------------------------------------------------------
# (a) registry-driven block ordering
# ---------------------------------------------------------------------------
def test_families_come_from_registry(corr_result):
    # Arrange
    families = set(corr_result["families"])
    # Act
    expected = {"pac_z_nan", "catch22"}
    # Assert
    assert families == expected


def test_unknown_name_falls_into_other_block(block_X, pac_names):
    # Arrange
    names = pac_names + [
        "nanq25",
        "not_a_registered_feature",
        "CO_f1ecac",
        "CO_FirstMin_ac",
    ]
    # Act
    result = feature_correlation(block_X, names)
    # Assert
    assert "other" in result["families"]


def test_block_boundaries_cover_all_features(corr_result, names):
    # Arrange
    boundaries = corr_result["block_boundaries"]
    # Act
    covered = sum(hi - lo for lo, hi in boundaries.values())
    # Assert
    assert covered == len(names)


def test_ordered_families_are_contiguous(corr_result):
    # Arrange
    ordered_families = corr_result["ordered_families"]
    # Act
    runs = [
        fam
        for i, fam in enumerate(ordered_families)
        if i == 0 or fam != ordered_families[i - 1]
    ]
    # Assert
    assert len(runs) == len(set(ordered_families))


def test_ordered_names_are_a_permutation(corr_result, names):
    # Arrange
    ordered_names = corr_result["ordered_names"]
    # Act
    same_set = set(ordered_names) == set(names)
    # Assert
    assert same_set


def test_block_order_false_preserves_input_order(block_X, names):
    # Arrange
    result = feature_correlation(block_X, names, block_order=False)
    # Act
    ordered_names = result["ordered_names"]
    # Assert
    assert ordered_names == list(names)


# ---------------------------------------------------------------------------
# (b) within-block vs cross-block summary
# ---------------------------------------------------------------------------
def test_summary_has_within_all_row(block_X, names):
    # Arrange
    summary = redundancy_summary(block_X, names)
    # Act
    scopes = set(summary["scope"])
    # Assert
    assert "within_all" in scopes


def test_summary_has_cross_row(block_X, names):
    # Arrange
    summary = redundancy_summary(block_X, names)
    # Act
    scopes = set(summary["scope"])
    # Assert
    assert "cross" in scopes


def test_summary_within_exceeds_cross(block_X, names):
    # Arrange
    summary = redundancy_summary(block_X, names).set_index("scope")
    # Act
    within = summary.loc["within_all", "mean_abs_r"]
    cross = summary.loc["cross", "mean_abs_r"]
    # Assert
    assert within > cross


def test_summary_per_family_rows_present(block_X, names):
    # Arrange
    summary = redundancy_summary(block_X, names)
    # Act
    families = set(summary[summary["scope"] == "within_family"]["family"])
    # Assert
    assert families == {"pac_z_nan", "catch22"}


def test_summary_pair_counts_are_positive(block_X, names):
    # Arrange
    summary = redundancy_summary(block_X, names).set_index("scope")
    # Act
    cross_pairs = summary.loc["cross", "n_pairs"]
    # Assert
    assert cross_pairs > 0


# ---------------------------------------------------------------------------
# (c) PCA loadings
# ---------------------------------------------------------------------------
def test_pca_loadings_shape_matches_features(block_X, names):
    # Arrange
    pytest.importorskip("sklearn")
    result = pca_loadings(block_X, names, n_components=3)
    # Act
    shape = result["loadings"].shape
    # Assert
    assert shape == (3, len(names))


def test_pca_loadings_columns_are_feature_names(block_X, names):
    # Arrange
    pytest.importorskip("sklearn")
    result = pca_loadings(block_X, names, n_components=2)
    # Act
    columns = list(result["loadings"].columns)
    # Assert
    assert columns == list(names)


def test_pca_explained_variance_sums_at_most_one(block_X, names):
    # Arrange
    pytest.importorskip("sklearn")
    result = pca_loadings(block_X, names)
    # Act
    total = float(result["explained_variance_ratio"].sum())
    # Assert
    assert total <= 1.0 + 1e-9


def test_pca_explained_variance_is_non_negative(block_X, names):
    # Arrange
    pytest.importorskip("sklearn")
    result = pca_loadings(block_X, names)
    # Act
    minimum = float(result["explained_variance_ratio"].min())
    # Assert
    assert minimum >= 0.0


def test_pca_result_carries_variance_caveat(block_X, names):
    # Arrange
    pytest.importorskip("sklearn")
    result = pca_loadings(block_X, names)
    # Act
    caveat = result["caveat"].lower()
    # Assert
    assert "not" in caveat and "predict" in caveat


def test_pca_n_components_clamped_to_max(block_X, names):
    # Arrange
    pytest.importorskip("sklearn")
    result = pca_loadings(block_X, names, n_components=999)
    # Act
    n_components = result["n_components"]
    # Assert
    assert n_components == min(block_X.shape[0], len(names))


# ---------------------------------------------------------------------------
# (d) per-group correlation + difference matrix
# ---------------------------------------------------------------------------
@pytest.fixture
def groups(block_X):
    # Split rows into two contiguous windows (e.g. preictal vs ictal).
    n = block_X.shape[0]
    labels = np.zeros(n, dtype=bool)
    labels[n // 2:] = True
    return labels


def test_group_result_has_matrix_per_group(block_X, names, groups):
    # Arrange
    result = correlation_by_group(block_X, names, groups)
    # Act
    n_matrices = len(result["matrices"])
    # Assert
    assert n_matrices == 2


def test_group_difference_matrix_shape(block_X, names, groups):
    # Arrange
    result = correlation_by_group(block_X, names, groups)
    # Act
    shape = result["difference"].shape
    # Assert
    assert shape == (len(names), len(names))


def test_identical_groups_give_zero_difference(block_X, names):
    # Arrange
    n = block_X.shape[0]
    labels = np.arange(n) % 2  # interleaved, identical distributions
    stacked = np.vstack([block_X, block_X])
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    result = correlation_by_group(stacked, names, labels)
    # Act
    max_abs_diff = np.abs(result["difference"].to_numpy()).max()
    # Assert
    assert max_abs_diff < 1e-9


def test_group_difference_is_none_for_three_groups(block_X, names):
    # Arrange
    n = block_X.shape[0]
    labels = np.arange(n) % 3
    # Act
    result = correlation_by_group(block_X, names, labels)
    # Assert
    assert result["difference"] is None


def test_group_length_mismatch_raises(block_X, names):
    # Arrange
    bad_groups = np.zeros(block_X.shape[0] + 1, dtype=bool)
    # Act
    # Assert
    with pytest.raises(ValueError):
        correlation_by_group(block_X, names, bad_groups)


def test_singleton_group_raises(block_X, names):
    # Arrange
    labels = np.zeros(block_X.shape[0], dtype=int)
    labels[0] = 1  # one group has a single sample
    # Act
    # Assert
    with pytest.raises(ValueError):
        correlation_by_group(block_X, names, labels)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
