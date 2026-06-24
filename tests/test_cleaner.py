import pandas as pd

from src.cleaner import CleaningConfig, apply_cleaning


def test_apply_cleaning_does_not_mutate_original_dataframe():
    df = pd.DataFrame({"a": [1, 1], "b": [None, 2]})
    original = df.copy(deep=True)

    apply_cleaning(df, CleaningConfig(drop_duplicates=True, missing_strategy="median"))

    pd.testing.assert_frame_equal(df, original)


def test_apply_cleaning_removes_duplicates_and_constant_columns():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "constant": ["same", "same", "same"],
            "value": [10, 10, 20],
        }
    )

    cleaned, audit = apply_cleaning(
        df,
        CleaningConfig(drop_duplicates=True, drop_constant_columns=True),
    )

    assert cleaned.shape == (2, 2)
    assert "constant" not in cleaned.columns
    assert audit.duplicates_removed == 1
    assert audit.columns_removed == ["constant"]


def test_apply_cleaning_fills_numeric_missing_with_median():
    df = pd.DataFrame({"value": [10.0, None, 30.0], "label": ["a", "b", "c"]})

    cleaned, audit = apply_cleaning(df, CleaningConfig(missing_strategy="median"))

    assert cleaned["value"].isna().sum() == 0
    assert cleaned.loc[1, "value"] == 20.0
    assert cleaned["label"].isna().sum() == 0
    assert audit.missing_values_filled == 1


def test_apply_cleaning_drops_rows_with_missing_values():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})

    cleaned, audit = apply_cleaning(df, CleaningConfig(missing_strategy="drop_rows"))

    assert cleaned.shape == (1, 2)
    assert audit.rows_removed_missing == 2
    assert audit.missing_values_after == 0