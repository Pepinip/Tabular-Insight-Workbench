"""Report generation utilities."""

from __future__ import annotations


def _correlation_label(value: float) -> str:
    abs_val = abs(value)
    direction = "positiva" if value > 0 else "negativa"
    if abs_val >= 0.7:
        strength = "fuerte"
    elif abs_val >= 0.4:
        strength = "moderada"
    else:
        strength = "debil"
    return f"{strength} {direction}"


def generate_executive_summary(report, model_result=None) -> str:
    lines = [
        "## Resumen ejecutivo", "",
        f"**Variable objetivo analizada:** `{report.target}`",
        f"**Variables predictoras seleccionadas:** {len(report.features)}", "",
        "### Correlaciones con la variable objetivo",
    ]

    if report.top_correlations.empty:
        lines.append("No se pudieron calcular correlaciones.")
    else:
        for _, row in report.top_correlations.head(3).iterrows():
            strength = _correlation_label(row["correlation"])
            lines.append(
                f"- **{row['feature']}**: correlacion {strength} ({row['correlation']:.2f})"
            )
        lines.append("")
        lines.append(
            "> ⚠️ Correlacion no implica causalidad. Estas relaciones son descriptivas, no explicativas."
        )

    if model_result is not None:
        lines += ["", "### Resultados del modelo de regresion lineal"]

        # Identifier exclusion notice
        if model_result.excluded_identifiers:
            lines.append(
                f"> 🔍 Variables excluidas por ser identificadores: "
                f"`{'`, `'.join(model_result.excluded_identifiers)}`. "
                f"No aportan valor predictivo real."
            )

        # Leakage warnings
        if model_result.leakage_warnings:
            for feat, corr in model_result.leakage_warnings:
                lines.append(
                    f"> 🚨 **Alerta de fuga de datos (Target Leakage):** `{feat}` "
                    f"tiene correlacion {corr} con el target. "
                    f"Es posible que esta variable contenga informacion del futuro o sea derivada del target. "
                    f"Considera excluirla del modelo."
                )

        lines += [
            f"- **MAE:** {model_result.mae} — error promedio absoluto",
            f"- **RMSE:** {model_result.rmse} — penaliza errores grandes",
            f"- **R²:** {model_result.r2:.4f} — varianza explicada en el conjunto de prueba",
            f"- **R² ajustado:** {model_result.r2_adjusted} — penaliza predictoras innecesarias",
            f"- **Train:** {model_result.n_train} filas | **Test:** {model_result.n_test} filas",
        ]

        if model_result.small_dataset:
            lines.append(
                f"- **Validacion cruzada (k-fold) R²:** "
                f"{model_result.cv_r2_mean} ± {model_result.cv_r2_std} "
                f"— mas confiable que el R² simple para datasets pequeños"
            )

        lines.append("")

        if model_result.low_performance:
            lines.append(
                "> ⚠️ R² < 0.30 — bajo desempeño. Considera revisar variables o usar otro modelo."
            )
        else:
            lines.append("> ✅ Desempeño aceptable para regresion lineal basica.")

        lines += [
            "", "### Limitaciones del modelo",
            "- Solo se usaron variables numericas como predictoras.",
            "- El modelo asume relaciones lineales entre variables.",
            "- Dataset pequeño puede no generalizar a datos nuevos.",
            "- Variables categoricas como `region` o `category` fueron ignoradas.",
            "- No se aplico regularizacion (Ridge/Lasso) ni seleccion de variables.",
        ]

    return "\n".join(lines)