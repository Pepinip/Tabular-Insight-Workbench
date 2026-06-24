"""Auditable data cleaning utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

MissingStrategy = Literal["none", "drop_rows", "mean", "median", "mode"]


@dataclass(frozen=True)
class CleaningConfig:
    """User-selected cleaning rules."""

    drop_duplicates: bool = False
    drop_constant_columns: bool = False
    missing_strategy: MissingStrategy = "none"


@dataclass(frozen=True)
class CleaningAudit:
    """Impact report for a cleaning operation."""

    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    duplicates_removed: int = 0
    columns_removed: list[str] = field(default_factory=list)
    rows_removed_missing: int = 0
    missing_values_before: int = 0
    missing_values_after: int = 0
    missing_values_filled: int = 0
    actions: list[str] = field(default_factory=list)


def _constant_columns(df: pd.DataFrame) -> list[str]:
    return df.nunique(dropna=True)[lambda values: values <= 1].index.tolist()


def _fill_missing_with_mode(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = df.copy()
    missing_before = int(cleaned.isna().sum().sum())

    for column in cleaned.columns:
        if not cleaned[column].isna().any():
            continue

        mode_values = cleaned[column].mode(dropna=True)
        if mode_values.empty:
            continue
        cleaned[column] = cleaned[column].fillna(mode_values.iloc[0])

    missing_after = int(cleaned.isna().sum().sum())
    return cleaned, missing_before - missing_after


def _fill_numeric_missing(df: pd.DataFrame, strategy: Literal["mean", "median"]) -> tuple[pd.DataFrame, int]:
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    missing_before = int(cleaned[numeric_columns].isna().sum().sum()) if len(numeric_columns) else 0

    for column in numeric_columns:
        if not cleaned[column].isna().any():
            continue

        value = cleaned[column].mean() if strategy == "mean" else cleaned[column].median()
        if pd.isna(value):
            continue
        cleaned[column] = cleaned[column].fillna(value)

    missing_after = int(cleaned[numeric_columns].isna().sum().sum()) if len(numeric_columns) else 0
    return cleaned, missing_before - missing_after


def apply_cleaning(df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, CleaningAudit]:
    """Apply explicit user-selected cleaning rules without mutating the source DataFrame."""
    cleaned = df.copy()
    rows_before, columns_before = cleaned.shape
    missing_before = int(cleaned.isna().sum().sum())
    duplicates_removed = 0
    rows_removed_missing = 0
    missing_values_filled = 0
    columns_removed: list[str] = []
    actions: list[str] = []

    if config.drop_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().copy()
        duplicates_removed = before - len(cleaned)
        actions.append("drop_duplicates")

    if config.drop_constant_columns:
        columns_removed = _constant_columns(cleaned)
        if columns_removed:
            cleaned = cleaned.drop(columns=columns_removed).copy()
        actions.append("drop_constant_columns")

    if config.missing_strategy == "drop_rows":
        before = len(cleaned)
        cleaned = cleaned.dropna().copy()
        rows_removed_missing = before - len(cleaned)
        actions.append("drop_rows_with_missing_values")
    elif config.missing_strategy in {"mean", "median"}:
        cleaned, missing_values_filled = _fill_numeric_missing(cleaned, config.missing_strategy)
        actions.append(f"fill_numeric_missing_with_{config.missing_strategy}")
    elif config.missing_strategy == "mode":
        cleaned, missing_values_filled = _fill_missing_with_mode(cleaned)
        actions.append("fill_missing_with_mode")
    else:
        actions.append("missing_values_unchanged")

    rows_after, columns_after = cleaned.shape
    missing_after = int(cleaned.isna().sum().sum())

    return cleaned, CleaningAudit(
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=columns_before,
        columns_after=columns_after,
        duplicates_removed=duplicates_removed,
        columns_removed=columns_removed,
        rows_removed_missing=rows_removed_missing,
        missing_values_before=missing_before,
        missing_values_after=missing_after,
        missing_values_filled=missing_values_filled,
        actions=actions,
    )