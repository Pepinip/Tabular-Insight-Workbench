"""Tests for src/modeling.py"""



import pandas as pd
import pytest
import matplotlib.pyplot as plt

from src.modeling import (
    ModelResult,
    compute_adjusted_r2,
    detect_leakage,
    is_likely_identifier,
    run_linear_regression,
    validate_inputs,
    plot_predictions,
    plot_residuals,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "sales":    [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                     150, 250, 350, 450, 550, 650, 750, 850, 950, 1050],
        "profit":   [10,  20,  30,  40,  50,  60,  70,  80,  90,  100,
                     15,  25,  35,  45,  55,  65,  75,  85,  95,  105],
        "discount": [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2,
                     0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2],
        "quantity": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                     1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "order_id": list(range(1001, 1021)),
    })


def test_is_likely_identifier_detects_id_column(sample_df):
    assert is_likely_identifier("order_id", sample_df) is True


def test_is_likely_identifier_does_not_flag_sales(sample_df):
    col_lower = "sales".lower()
    has_id_pattern = any(
        p in col_lower
        for p in ["id", "code", "cod", "key", "index", "num", "numero", "folio"]
    )
    assert has_id_pattern is False


def test_is_likely_identifier_detects_unique_columns(sample_df):
    assert is_likely_identifier("order_id", sample_df) is True


def test_detect_leakage_finds_high_correlation(sample_df):
    leakage = detect_leakage(sample_df, "sales", ["profit", "discount"], threshold=0.85)
    leakage_features = [f for f, _ in leakage]
    assert "profit" in leakage_features


def test_detect_leakage_does_not_flag_low_correlation(sample_df):
    leakage = detect_leakage(sample_df, "sales", ["discount"], threshold=0.85)
    leakage_features = [f for f, _ in leakage]
    assert "discount" not in leakage_features


def test_compute_adjusted_r2_penalizes_extra_features():
    r2 = 0.95
    adj_standard = compute_adjusted_r2(r2, n=20, k=1)
    adj_many = compute_adjusted_r2(r2, n=20, k=5)
    assert adj_standard > adj_many


def test_compute_adjusted_r2_returns_nan_when_insufficient_data():
    import math
    result = compute_adjusted_r2(0.9, n=2, k=5)
    assert math.isnan(result)


def test_validate_inputs_returns_none_when_valid(sample_df):
    error = validate_inputs(sample_df, "sales", ["profit", "discount"])
    assert error is None


def test_validate_inputs_rejects_missing_target(sample_df):
    error = validate_inputs(sample_df, "nonexistent", ["profit"])
    assert error is not None
    assert "nonexistent" in error


def test_validate_inputs_rejects_non_numeric_target(sample_df):
    df = sample_df.copy()
    df["label"] = "text"
    error = validate_inputs(df, "label", ["profit"])
    assert error is not None


def test_validate_inputs_rejects_no_numeric_features(sample_df):
    df = sample_df.copy()
    df["category"] = "A"
    error = validate_inputs(df, "sales", ["category"])
    assert error is not None


def test_run_linear_regression_excludes_identifiers(sample_df):
    result = run_linear_regression(sample_df, "sales", ["order_id", "profit", "discount"])
    assert "order_id" in result.excluded_identifiers
    assert "order_id" not in result.features


def test_run_linear_regression_does_not_mutate_df(sample_df):
    original = sample_df.copy(deep=True)
    run_linear_regression(sample_df, "sales", ["profit", "discount"])
    pd.testing.assert_frame_equal(sample_df, original)


def test_run_linear_regression_returns_model_result(sample_df):
    result = run_linear_regression(sample_df, "sales", ["profit", "discount"])
    assert isinstance(result, ModelResult)
    assert result.n_train + result.n_test == result.n_total


def test_run_linear_regression_metrics_in_valid_range(sample_df):
    result = run_linear_regression(sample_df, "sales", ["profit", "discount"])
    assert result.mae >= 0
    assert result.rmse >= 0
    assert result.r2 <= 1.0


def test_run_linear_regression_small_dataset_uses_cv(sample_df):
    import math
    result = run_linear_regression(sample_df, "sales", ["profit", "discount"])
    assert result.small_dataset is True
    assert not math.isnan(result.cv_r2_mean)


def test_run_linear_regression_predictions_shape(sample_df):
    result = run_linear_regression(sample_df, "sales", ["profit", "discount"])
    assert "y_real" in result.predictions.columns
    assert "y_pred" in result.predictions.columns
    assert len(result.predictions) == result.n_test


def test_plot_predictions_returns_figure(sample_df):
    result = run_linear_regression(sample_df, "sales", ["profit", "discount"])
    fig = plot_predictions(result)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_residuals_returns_figure(sample_df):
    result = run_linear_regression(sample_df, "sales", ["profit", "discount"])
    fig = plot_residuals(result)
    assert isinstance(fig, plt.Figure)
    plt.close("all")