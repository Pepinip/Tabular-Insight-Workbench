import pandas as pd

from src.profiler import build_profile


def test_build_profile_detects_basic_quality_metrics():
    df = pd.DataFrame(
        {
            "a": [1, 1, None],
            "b": ["x", "x", "y"],
            "constant": ["same", "same", "same"],
        }
    )

    profile = build_profile(df)

    assert profile.rows == 3
    assert profile.columns == 3
    assert profile.missing_cells == 1
    assert "a" in profile.numeric_columns
    assert "constant" in profile.constant_columns