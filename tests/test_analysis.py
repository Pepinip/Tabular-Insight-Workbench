"""Tests for src/analysis.py"""


import pandas as pd
import pytest
import matplotlib.pyplot as plt

from src.analysis import (
    build_exploratory_report,
    compute_correlation_matrix,
    compute_target_correlations,
    get_numeric_columns,
    plot_correlation_heatmap,
    plot_histograms,
    plot_scatter,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "sales":    [100.0, 200.0, 300.0, 400.0, 500.0],
        "profit":   [10.0,  20.0,  30.0,  40.0,  50.0],
        "discount": [0.1,   0.2,   0.1,   0.3,   0.2],
        "quantity": [1, 2, 3, 4, 5],
        "region":   ["North", "South", "East", "West", "North"],
    })


def test_get_numeric_columns_returns_only_numeric(sample_df):
    numeric = get_numeric_columns(sample_df)
    assert "sales" in numeric
    assert "profit" in numeric
    assert "region" not in numeric


def test_get_numeric_columns_excludes_strings(sample_df):
    numeric = get_numeric_columns(sample_df)
    for col in numeric:
        assert pd.api.types.is_numeric_dtype(sample_df[col])


def test_compute_correlation_matrix_shape(sample_df):
    corr = compute_correlation_matrix(sample_df)
    numeric_count = len(get_numeric_columns(sample_df))
    assert corr.shape == (numeric_count, numeric_count)


def test_compute_correlation_matrix_diagonal_is_one(sample_df):
    corr = compute_correlation_matrix(sample_df)
    for col in corr.columns:
        assert abs(corr.loc[col, col] - 1.0) < 1e-10


def test_compute_target_correlations_returns_sorted(sample_df):
    corr = compute_target_correlations(sample_df, "sales")
    assert not corr.empty
    abs_values = corr["correlation"].abs().tolist()
    assert abs_values == sorted(abs_values, reverse=True)


def test_compute_target_correlations_excludes_target(sample_df):
    corr = compute_target_correlations(sample_df, "sales")
    assert "sales" not in corr["feature"].values


def test_compute_target_correlations_invalid_target(sample_df):
    corr = compute_target_correlations(sample_df, "nonexistent")
    assert corr.empty


def test_build_exploratory_report_structure(sample_df):
    report = build_exploratory_report(sample_df, "sales", ["profit", "discount"])
    assert report.target == "sales"
    assert "profit" in report.features
    assert not report.correlation_matrix.empty
    assert not report.top_correlations.empty


def test_build_exploratory_report_does_not_mutate_df(sample_df):
    original = sample_df.copy(deep=True)
    build_exploratory_report(sample_df, "sales", ["profit"])
    pd.testing.assert_frame_equal(sample_df, original)


def test_plot_correlation_heatmap_returns_figure(sample_df):
    corr = compute_correlation_matrix(sample_df)
    fig = plot_correlation_heatmap(corr)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_histograms_returns_figure(sample_df):
    numeric = get_numeric_columns(sample_df)
    fig = plot_histograms(sample_df, numeric)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_scatter_returns_figure(sample_df):
    fig = plot_scatter(sample_df, "sales", ["profit", "discount"])
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_histograms_empty_columns(sample_df):
    fig = plot_histograms(sample_df, [])
    assert isinstance(fig, plt.Figure)
    plt.close("all")