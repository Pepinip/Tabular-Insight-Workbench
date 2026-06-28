"""Streamlit entrypoint for Tabular Insight Workbench."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from src.analysis import (
    build_exploratory_report,
    get_numeric_columns,
    plot_correlation_heatmap,
    plot_histograms,
    plot_scatter,
)
from src.cleaner import CleaningConfig, apply_cleaning
from src.data_loader import DatasetLoadError, load_tabular_file
from src.logger import logger
from src.modeling import plot_predictions, plot_residuals, run_linear_regression, validate_inputs
from src.profiler import build_profile
from src.reporting import generate_docx_report, generate_executive_summary, generate_txt_report

st.set_page_config(page_title="Tabular Insight Workbench", layout="wide")

os.makedirs("outputs/figures", exist_ok=True)


@st.cache_data(show_spinner=False)
def load_dataset_cached(filename: str, file_bytes: bytes):
    return load_tabular_file(filename, file_bytes)


def render_profile_metrics(profile) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Filas", f"{profile.rows:,}")
    col2.metric("Columnas", f"{profile.columns:,}")
    col3.metric("Duplicados", f"{profile.duplicate_rows:,}")
    col4.metric("Celdas nulas", f"{profile.missing_cells:,}")
    col5.metric("Nulos %", f"{profile.missing_cells_pct:.2f}%")


def render_quality_warnings(profile) -> None:
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
        st.warning("\n".join(f"- {w}" for w in warnings))
    else:
        st.success("No se detectaron alertas basicas de calidad.")


def render_cleaning_audit(audit) -> None:
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


def render_analysis_section(clean_df: pd.DataFrame, audit, filename: str) -> None:
    st.divider()
    st.header("Analisis exploratorio y modelo")

    numeric_cols = get_numeric_columns(clean_df)

    if len(numeric_cols) < 2:
        st.warning("Se necesitan al menos 2 columnas numericas para el analisis.")
        return

    col_left, col_right = st.columns(2)
    with col_left:
        target = st.selectbox(
            "Variable objetivo (target)",
            options=numeric_cols,
            help="La variable que quieres predecir o analizar.",
        )
    with col_right:
        default_features = [c for c in numeric_cols if c != target]
        features = st.multiselect(
            "Variables predictoras",
            options=default_features,
            default=default_features,
            help="Variables que se usaran para predecir el target.",
        )

    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False
    if "last_target" not in st.session_state or st.session_state.last_target != target:
        st.session_state.analysis_running = False
        st.session_state.last_target = target
    if "last_features" not in st.session_state or st.session_state.last_features != features:
        st.session_state.analysis_running = False
        st.session_state.last_features = features

    if st.button("Ejecutar analisis y modelo", type="primary"):
        st.session_state.analysis_running = True
        logger.info(f"Analisis ejecutado | target={target} | features={features}")

    if not st.session_state.analysis_running:
        st.info("Selecciona el target y las predictoras, luego presiona el boton.")
        return

    if not features:
        st.error("Selecciona al menos una variable predictora.")
        return

    report = build_exploratory_report(clean_df, target, features)

    tab_corr, tab_hist, tab_scatter, tab_model, tab_report = st.tabs([
        "Correlacion", "Distribuciones", "Dispersion", "Modelo", "Reporte"
    ])

    with tab_corr:
        st.subheader("Matriz de correlacion")
        fig_heatmap = plot_correlation_heatmap(report.correlation_matrix)
        st.pyplot(fig_heatmap)
        fig_heatmap.savefig("outputs/figures/correlation_heatmap.png", dpi=100, bbox_inches="tight")
        if not report.top_correlations.empty:
            st.subheader(f"Correlaciones con '{target}'")
            st.dataframe(report.top_correlations, use_container_width=True, hide_index=True)

    with tab_hist:
        st.subheader("Distribucion de variables numericas")
        fig_hist = plot_histograms(clean_df, report.numeric_columns)
        st.pyplot(fig_hist)
        fig_hist.savefig("outputs/figures/histograms.png", dpi=100, bbox_inches="tight")

    with tab_scatter:
        st.subheader(f"Dispersion: predictoras vs {target}")
        fig_scatter = plot_scatter(clean_df, target, features)
        st.pyplot(fig_scatter)
        fig_scatter.savefig("outputs/figures/scatter_plots.png", dpi=100, bbox_inches="tight")

    with tab_model:
        st.subheader("Regresion lineal")
        error_msg = validate_inputs(clean_df, target, features)
        if error_msg:
            logger.warning(f"Validacion fallida: {error_msg}")
            st.error(error_msg)
        else:
            try:
                result = run_linear_regression(clean_df, target, features)
                logger.info(
                    f"Modelo entrenado | target={result.target} | "
                    f"R2={result.r2} | R2_adj={result.r2_adjusted} | "
                    f"MAE={result.mae} | RMSE={result.rmse} | "
                    f"features={result.features}"
                )
            except ValueError as e:
                logger.error(f"Error en regresion lineal: {e}")
                st.error(f"❌ Error al entrenar el modelo: {e}")
                st.stop()

            if result.excluded_identifiers:
                logger.info(f"Identificadores excluidos del modelo: {result.excluded_identifiers}")
                st.info(
                    f"🔍 Variables excluidas automaticamente por ser identificadores: "
                    f"`{'`, `'.join(result.excluded_identifiers)}`. "
                    f"No aportan valor predictivo real."
                )

            if result.leakage_warnings:
                for feat, corr in result.leakage_warnings:
                    logger.warning(f"Target Leakage detectado: {feat} correlaciona {corr} con {target}")
                    st.error(
                        f"🚨 **Target Leakage detectado:** `{feat}` correlaciona {corr} "
                        f"con `{target}`. Considera excluirla del modelo."
                    )

            if result.low_performance:
                st.warning(f"⚠️ R² = {result.r2:.4f} — desempeño bajo.")
            else:
                st.success(f"✅ R² = {result.r2:.4f} — desempeño aceptable.")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{result.mae:,.4f}")
            col2.metric("RMSE", f"{result.rmse:,.4f}")
            col3.metric("R²", f"{result.r2:.4f}")
            col4.metric("R² ajustado", f"{result.r2_adjusted:.4f}")

            if result.small_dataset:
                st.warning(
                    f"⚠️ Dataset pequeño ({result.n_total} filas). "
                    f"Validacion cruzada R²: {result.cv_r2_mean} ± {result.cv_r2_std}"
                )

            st.caption(
                f"Entrenamiento: {result.n_train} filas | "
                f"Prueba: {result.n_test} filas | "
                f"Predictoras: {', '.join(result.features)}"
            )

            fig_pred = plot_predictions(result)
            st.pyplot(fig_pred)
            fig_pred.savefig("outputs/figures/predictions.png", dpi=100, bbox_inches="tight")

            st.subheader("Grafico de residuos")
            st.caption("Residuos aleatorios alrededor de 0 indican un modelo bien ajustado.")
            fig_resid = plot_residuals(result)
            st.pyplot(fig_resid)
            fig_resid.savefig("outputs/figures/residuals.png", dpi=100, bbox_inches="tight")

            st.subheader("Predicciones vs valores reales")
            st.dataframe(result.predictions.head(20), use_container_width=True, hide_index=True)

    with tab_report:
        model_result = None
        if not validate_inputs(clean_df, target, features):
            try:
                model_result = run_linear_regression(clean_df, target, features)
            except ValueError:
                pass

        clean_profile = build_profile(clean_df)
        summary = generate_executive_summary(report, model_result)
        st.markdown(summary)

        st.divider()
        st.subheader("Descargar reporte")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        col1, col2 = st.columns(2)

        with col1:
            txt_content = generate_txt_report(
                filename=filename,
                profile=clean_profile,
                audit=audit,
                report=report,
                model_result=model_result,
            )
            st.download_button(
                label="⬇️ Descargar TXT",
                data=txt_content,
                file_name=f"reporte_{target}_{timestamp}.txt",
                mime="text/plain",
            )

        with col2:
            try:
                docx_bytes = generate_docx_report(
                    filename=filename,
                    profile=clean_profile,
                    audit=audit,
                    report=report,
                    model_result=model_result,
                )
                st.download_button(
                    label="⬇️ Descargar DOCX",
                    data=docx_bytes,
                    file_name=f"reporte_{target}_{timestamp}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            except Exception as e:
                logger.error(f"Error generando DOCX: {e}")
                st.error(f"Error generando DOCX: {e}")


def main() -> None:
    st.title("Tabular Insight Workbench")
    st.caption("Carga, perfila, limpia y evalua datasets tabulares con reglas explicitas.")

    if "cleaning_config" not in st.session_state:
        st.session_state.cleaning_config = CleaningConfig()
    if "last_filename" not in st.session_state:
        st.session_state.last_filename = None

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
            format_func=lambda v: {
                "none": "No hacer nada",
                "drop_rows": "Eliminar filas con nulos",
                "mean": "Rellenar numericas con media",
                "median": "Rellenar numericas con mediana",
                "mode": "Rellenar con moda",
            }[v],
        )
        apply_rules = st.button("Aplicar limpieza", type="primary")

    if uploaded_file is None:
        st.info("Sube un CSV o Excel para iniciar el diagnostico del dataset.")
        return

    file_bytes = uploaded_file.getvalue()

    try:
        loaded = load_dataset_cached(uploaded_file.name, file_bytes)
        logger.info(
            f"Archivo cargado: {uploaded_file.name} | "
            f"{loaded.size_mb:.2f} MB | "
            f"hash: {loaded.dataset_hash[:12]}"
        )
    except DatasetLoadError as exc:
        logger.error(f"Error cargando archivo '{uploaded_file.name}': {exc}")
        st.error(str(exc))
        return

    raw_df = loaded.dataframe
    raw_profile = build_profile(raw_df)

    if uploaded_file.name != st.session_state.last_filename:
        st.session_state.cleaning_config = CleaningConfig()
        st.session_state.last_filename = uploaded_file.name
        if "analysis_running" in st.session_state:
            st.session_state.analysis_running = False

    if apply_rules:
        st.session_state.cleaning_config = CleaningConfig(
            drop_duplicates=drop_duplicates,
            drop_constant_columns=drop_constant_columns,
            missing_strategy=missing_strategy,
        )

    clean_df, audit = apply_cleaning(raw_df, st.session_state.cleaning_config)
    logger.info(
        f"Limpieza aplicada: {audit.actions} | "
        f"filas: {audit.rows_before} -> {audit.rows_after} | "
        f"columnas: {audit.columns_before} -> {audit.columns_after}"
    )

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

    render_analysis_section(clean_df, audit, loaded.filename)


if __name__ == "__main__":
    main()