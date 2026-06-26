"""Exploratory analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class ExploratoryReport:
    numeric_columns: list[str]
    correlation_matrix: pd.DataFrame
    top_correlations: pd.DataFrame
    target: str
    features: list[str]


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    return numeric.corr(method="pearson")


def compute_target_correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    corr_matrix = compute_correlation_matrix(df)
    if target not in corr_matrix.columns:
        return pd.DataFrame()
    correlations = (
        corr_matrix[[target]]
        .drop(index=target, errors="ignore")
        .rename(columns={target: "correlation"})
        .assign(abs_correlation=lambda x: x["correlation"].abs())
        .sort_values("abs_correlation", ascending=False)
        .drop(columns="abs_correlation")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return correlations


def build_exploratory_report(df: pd.DataFrame, target: str, features: list[str]) -> ExploratoryReport:
    numeric_columns = get_numeric_columns(df)
    correlation_matrix = compute_correlation_matrix(df)
    top_correlations = compute_target_correlations(df, target)
    return ExploratoryReport(
        numeric_columns=numeric_columns,
        correlation_matrix=correlation_matrix,
        top_correlations=top_correlations,
        target=target,
        features=features,
    )


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlacion", fontsize=14, pad=12)
    fig.tight_layout()
    return fig


def plot_histograms(df: pd.DataFrame, numeric_columns: list[str]) -> plt.Figure:
    n = len(numeric_columns)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin columnas numericas", ha="center", va="center")
        return fig
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = [axes] if n == 1 else axes.flatten()
    for i, col in enumerate(numeric_columns):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribucion de variables numericas", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, target: str, features: list[str]) -> plt.Figure:
    valid = [f for f in features if f in df.columns and f != target]
    n = len(valid)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin variables predictoras validas", ha="center", va="center")
        return fig
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = [axes] if n == 1 else axes.flatten()
    for i, feat in enumerate(valid):
        axes[i].scatter(df[feat], df[target], alpha=0.6, color="steelblue", s=40)
        axes[i].set_xlabel(feat, fontsize=10)
        axes[i].set_ylabel(target, fontsize=10)
        axes[i].set_title(f"{feat} vs {target}", fontsize=11)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Dispersion vs {target}", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig
