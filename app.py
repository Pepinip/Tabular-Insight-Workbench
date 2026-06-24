"""Streamlit entrypoint for Tabular Insight Workbench."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.cleaner import CleaningConfig, apply_cleaning
from src.data_loader import DatasetLoadError, load_tabular_file
from src.profiler import build_profile


st.set_page_config(page_title="Tabular Insight Workbench", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset_cached(filename: str, file_bytes: bytes):
    """Cache parsing by file content and name."""
    return load_tabular_file(filename, file_bytes)


def render_profile_metrics(profile) -> None:
    """Render top-level dataset quality metrics."""
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Filas", f"{profile.rows:,}")
    col2.metric("Columnas", f"{profile.columns:,}")
    col3.metric("Duplicados", f"{profile.duplicate_rows:,}")
    col4.metric("Celdas nulas", f"{profile.missing_cells:,}")
    col5.metric("Nulos %", f"{profile.missing_cells_pct:.2f}%")


def render_quality_warnings(profile) -> None:
    """Render readable warnings based on profile results."""
    warnings = []
    if profile.duplicate_rows > 0:
        warnings.append(f"Se detectaron {profile.duplicate_rows:,} filas duplicadas.")
    if profile.missing_cells > 0:
        warnings.append(f"Se detectaron {profile.missing_cells:,} celdas con valores nulos.")
    if profile.constant_columns:
        warnings.append(
            "Columnas constantes: "
            f"{', '.join(profile.constant_columns)}. No aportan variacion para analisis o modelos."
        )
    if profile.high_cardinality_columns:
        warnings.append(
            "Columnas de alta cardinalidad: "
            f"{', '.join(profile.high_cardinality_columns)}. Revisar si son identificadores o texto libre."
        )

    if warnings:
        st.warning("\n".join(f"- {warning}" for warning in warnings))
    else:
        st.success("No se detectaron alertas basicas de calidad.")


def render_cleaning_audit(audit) -> None:
    """Render the audit trail for explicit cleaning actions."""
    st.subheader("Auditoria de limpieza")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas antes", f"{audit.rows_before:,}")
    col2.metric("Filas despues", f"{audit.rows_after:,}", delta=audit.rows_after - audit.rows_before)
    col3.metric("Columnas antes", f"{audit.columns_before:,}")
    col4.metric("Columnas despues", f"{audit.columns_after:,}", delta=audit.columns_after - audit.columns_before)

    audit_rows = [
        {"metrica": "Duplicados eliminados", "valor": audit.duplicates_removed},
        {"metrica": "Filas eliminadas por nulos", "valor": audit.rows_removed_missing},
        {"metrica": "Nulos antes", "valor": audit.missing_values_before},
        {"metrica": "Nulos despues", "valor": audit.missing_values_after},
        {"metrica": "Valores imputados", "valor": audit.missing_values_filled},
        {"metrica": "Columnas eliminadas", "valor": ", ".join(audit.columns_removed) or "Ninguna"},
        {"metrica": "Acciones aplicadas", "valor": ", ".join(audit.actions)},
    ]
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)


def render_dataset_tabs(raw_df: pd.DataFrame, clean_df: pd.DataFrame, profile) -> None:
    """Render raw and cleaned dataset inspection tabs."""
    tab_raw, tab_clean, tab_profile, tab_numeric = st.tabs(
        ["Datos originales", "Datos limpios", "Perfil de columnas", "Resumen numerico"]
    )

    with tab_raw:
        st.caption("raw_df: dataset original sin modificaciones.")
        st.dataframe(raw_df.head(50), use_container_width=True)

    with tab_clean:
        st.caption("clean_df: dataset derivado a partir de reglas explicitas.")
        st.dataframe(clean_df.head(50), use_container_width=True)

    with tab_profile:
        st.dataframe(profile.column_profile, use_container_width=True, hide_index=True)

    with tab_numeric:
        if profile.numeric_summary.empty:
            st.info("No hay columnas numericas para resumir.")
        else:
            st.dataframe(profile.numeric_summary, use_container_width=True)


def main() -> None:
    st.title("Tabular Insight Workbench")
    st.caption("Carga, perfila, limpia y evalua datasets tabulares con reglas explicitas.")

    with st.sidebar:
        st.header("Entrada")
        uploaded_file = st.file_uploader(
            "Sube un archivo CSV o Excel",
            type=["csv", "xlsx", "xls"],
        )
        st.caption("Limite MVP: 25 MB por archivo.")
        st.divider()

        st.header("Limpieza")
        drop_duplicates = st.checkbox("Eliminar duplicados")
        drop_constant_columns = st.checkbox("Eliminar columnas constantes")
        missing_strategy = st.selectbox(
            "Manejo de nulos",
            options=["none", "drop_rows", "mean", "median", "mode"],
            format_func=lambda value: {
                "none": "No hacer nada",
                "drop_rows": "Eliminar filas con nulos",
                "mean": "Rellenar numericas con media",
                "median": "Rellenar numericas con mediana",
                "mode": "Rellenar con moda",
            }[value],
        )
        apply_rules = st.button("Aplicar limpieza", type="primary")

    if uploaded_file is None:
        st.info("Sube un CSV o Excel para iniciar el diagnostico del dataset.")
        return

    file_bytes = uploaded_file.getvalue()

    try:
        loaded = load_dataset_cached(uploaded_file.name, file_bytes)
    except DatasetLoadError as exc:
        st.error(str(exc))
        return

    raw_df = loaded.dataframe
    raw_profile = build_profile(raw_df)

    config = CleaningConfig(
        drop_duplicates=drop_duplicates,
        drop_constant_columns=drop_constant_columns,
        missing_strategy=missing_strategy,
    )

    if apply_rules:
        clean_df, audit = apply_cleaning(raw_df, config)
    else:
        clean_df, audit = apply_cleaning(raw_df, CleaningConfig())

    clean_profile = build_profile(clean_df)

    st.subheader("Resumen del dataset")
    st.write(
        f"Archivo: `{loaded.filename}` | "
        f"Tamano: `{loaded.size_mb:.2f} MB` | "
        f"Hash: `{loaded.dataset_hash[:12]}`"
    )

    render_profile_metrics(clean_profile)
    render_quality_warnings(clean_profile)
    render_cleaning_audit(audit)
    render_dataset_tabs(raw_df, clean_df, clean_profile)

    with st.expander("Perfil original antes de limpieza"):
        render_profile_metrics(raw_profile)
        st.dataframe(raw_profile.column_profile, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()