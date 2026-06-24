"""Data profiling utilities for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    """Summary profile for a DataFrame."""

    rows: int
    columns: int
    duplicate_rows: int
    missing_cells: int
    missing_cells_pct: float
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    column_profile: pd.DataFrame
    numeric_summary: pd.DataFrame


def _detect_high_cardinality(df: pd.DataFrame, threshold: float = 0.8) -> list[str]:
    """Detect likely identifier/text columns with too many unique values."""
    result: list[str] = []
    if len(df) == 0:
        return result

    non_numeric_columns = df.select_dtypes(exclude=["number", "datetime", "datetimetz"]).columns
    for column in non_numeric_columns:
        ratio = df[column].nunique(dropna=True) / len(df)
        if ratio >= threshold:
            result.append(column)
    return result


def build_profile(df: pd.DataFrame) -> DatasetProfile:
    """Build a non-destructive quality profile for a DataFrame."""
    rows, columns = df.shape
    missing_by_column = df.isna().sum()
    missing_pct = (missing_by_column / rows * 100).round(2) if rows else missing_by_column
    unique_counts = df.nunique(dropna=True)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=["number", "datetime", "datetimetz"]).columns.tolist()
    constant_columns = unique_counts[unique_counts <= 1].index.tolist()
    high_cardinality_columns = _detect_high_cardinality(df)

    column_profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": missing_by_column.values,
            "missing_pct": missing_pct.values,
            "unique_count": unique_counts.values,
        }
    )

    numeric_summary = df[numeric_columns].describe().T if numeric_columns else pd.DataFrame()
    missing_cells = int(missing_by_column.sum())
    total_cells = rows * columns

    return DatasetProfile(
        rows=rows,
        columns=columns,
        duplicate_rows=int(df.duplicated().sum()),
        missing_cells=missing_cells,
        missing_cells_pct=round((missing_cells / total_cells * 100), 2) if total_cells else 0.0,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        column_profile=column_profile,
        numeric_summary=numeric_summary,
    )