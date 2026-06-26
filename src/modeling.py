"""Modeling utilities - linear regression with train/test split."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

IDENTIFIER_PATTERNS = ["id", "code", "cod", "key", "index", "num", "numero", "folio"]


def is_likely_identifier(col: str, df: pd.DataFrame) -> bool:
    col_lower = col.lower()
    if any(pattern in col_lower for pattern in IDENTIFIER_PATTERNS):
        return True
    if df[col].nunique() == len(df):
        return True
    return False


def detect_leakage(df: pd.DataFrame, target: str, features: list, threshold: float = 0.85) -> list:
    leakage = []
    for feat in features:
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            corr = abs(df[feat].corr(df[target]))
            if corr >= threshold:
                leakage.append((feat, round(corr, 2)))
    return leakage


def compute_adjusted_r2(r2: float, n: int, k: int) -> float:
    if n - k - 1 <= 0:
        return float("nan")
    return round(1 - (1 - r2) * (n - 1) / (n - k - 1), 4)


@dataclass
class ModelResult:
    mae: float
    rmse: float
    r2: float
    r2_adjusted: float
    cv_r2_mean: float
    cv_r2_std: float
    n_train: int
    n_test: int
    n_total: int
    target: str
    features: list
    excluded_identifiers: list
    leakage_warnings: list
    predictions: pd.DataFrame
    low_performance: bool
    small_dataset: bool


def validate_inputs(df: pd.DataFrame, target: str, features: list):
    if target not in df.columns:
        return f"La columna objetivo '{target}' no existe."
    if not pd.api.types.is_numeric_dtype(df[target]):
        return f"La columna '{target}' debe ser numerica."
    numeric_features = [
        f for f in features
        if f in df.columns and f != target and pd.api.types.is_numeric_dtype(df[f])
    ]
    if len(numeric_features) < 1:
        return "Se necesita al menos una variable predictora numerica."
    if len(df.dropna(subset=[target] + numeric_features)) < 10:
        return "Se necesitan al menos 10 filas sin nulos."
    return None


def run_linear_regression(
    df: pd.DataFrame,
    target: str,
    features: list,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    numeric_features = [
        f for f in features
        if f in df.columns and f != target and pd.api.types.is_numeric_dtype(df[f])
    ]

    excluded_identifiers = [f for f in numeric_features if is_likely_identifier(f, df)]
    clean_features = [f for f in numeric_features if f not in excluded_identifiers]

    # Si todas fueron excluidas, usar las originales sin filtro
    if not clean_features:
        clean_features = numeric_features
        excluded_identifiers = []

    leakage_warnings = detect_leakage(df, target, clean_features, threshold=0.85)

    subset = df[[target] + clean_features].dropna()

    if len(clean_features) == 0 or subset.empty or len(subset) < 2:
        raise ValueError(
            "No quedaron variables predictoras validas despues del filtrado. "
            "Selecciona otras variables o revisa el dataset."
        )

    X = subset[clean_features].values
    y = subset[target].values
    n_total = len(subset)
    k = len(clean_features)
    small_dataset = n_total < 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    r2_adjusted = compute_adjusted_r2(r2, len(X_test), k)

    if small_dataset and n_total >= 10:
        n_splits = min(5, n_total // 2)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
        cv_r2_mean = round(float(cv_scores.mean()), 4)
        cv_r2_std = round(float(cv_scores.std()), 4)
    else:
        cv_r2_mean = float("nan")
        cv_r2_std = float("nan")

    predictions = pd.DataFrame({"y_real": y_test, "y_pred": y_pred.round(2)})

    return ModelResult(
        mae=round(mae, 4), rmse=round(rmse, 4), r2=round(r2, 4),
        r2_adjusted=r2_adjusted, cv_r2_mean=cv_r2_mean, cv_r2_std=cv_r2_std,
        n_train=len(X_train), n_test=len(X_test), n_total=n_total,
        target=target, features=clean_features,
        excluded_identifiers=excluded_identifiers,
        leakage_warnings=leakage_warnings,
        predictions=predictions,
        low_performance=r2 < 0.3,
        small_dataset=small_dataset,
    )


def plot_predictions(result: ModelResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(result.predictions["y_real"], result.predictions["y_pred"],
               alpha=0.7, color="steelblue", s=50, label="Predicciones")
    mn = min(result.predictions["y_real"].min(), result.predictions["y_pred"].min())
    mx = max(result.predictions["y_real"].max(), result.predictions["y_pred"].max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel(f"Valor real ({result.target})")
    ax.set_ylabel(f"Valor predicho ({result.target})")
    ax.set_title("Predicciones vs Valores reales")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(result: ModelResult) -> plt.Figure:
    residuals = result.predictions["y_real"] - result.predictions["y_pred"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(result.predictions["y_pred"], residuals,
               alpha=0.7, color="steelblue", s=50)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"Valor predicho ({result.target})")
    ax.set_ylabel("Residuo (real - predicho)")
    ax.set_title("Grafico de residuos")
    fig.tight_layout()
    return fig