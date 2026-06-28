+# Tabular Insight Workbench

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-39%20passed-brightgreen)](tests/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit%20Cloud-ff4b4b?logo=streamlit)](https://tabular-insight-workbench.streamlit.app)

Herramienta web de analisis de datos tabulares que permite cargar datasets CSV o Excel y obtener en minutos un diagnostico completo de calidad, limpieza auditable, analisis exploratorio y un modelo predictivo basico — sin escribir una sola linea de codigo.

---

## Demo en vivo

**[tabular-insight-workbench.streamlit.app](https://tabular-insight-workbench.streamlit.app)**

---

## Que hace
CSV / Excel
↓
Carga y validacion (hash SHA-256, limite 25MB)
↓
Perfilado de calidad (nulos, duplicados, columnas constantes)
↓
Limpieza auditable (reglas explicitas + registro de impacto)
↓
Analisis exploratorio (correlacion, histogramas, scatter plots)
↓
Modelo de regresion lineal (MAE, RMSE, R², R² ajustado, CV k-fold)
↓
Reporte descargable (TXT y DOCX)
---

## Funcionalidades principales

### Semana 1 — Carga y perfilado
- Carga de archivos CSV, XLSX y XLS hasta 25MB
- Validacion de formato y hash SHA-256 por archivo
- Deteccion automatica de nulos, duplicados, columnas constantes y alta cardinalidad
- Vista previa, perfil por columna y resumen estadistico numerico

### Semana 2 — Limpieza auditable
- Eliminacion de duplicados
- Eliminacion de columnas constantes
- Manejo de nulos: eliminar filas, imputar media, mediana o moda
- Auditoria completa: filas/columnas antes y despues, acciones aplicadas
- `raw_df` nunca se modifica — `clean_df` siempre es una copia

### Semana 3 — Analisis exploratorio y modelo
- Matriz de correlacion con heatmap interactivo
- Histogramas de distribucion por variable numerica
- Scatter plots de predictoras vs target
- Deteccion automatica de identificadores (excluidos del modelo)
- Deteccion de Target Leakage (correlacion > 0.85 con el target)
- Regresion lineal con scikit-learn
- Metricas: MAE, RMSE, R², R² ajustado
- Validacion cruzada k-fold para datasets pequenos (< 100 filas)
- Grafico de residuos

### Semana 4 — Calidad de produccion
- 39 tests automatizados con pytest (100% passing)
- Logging en archivo con registro de cada sesion
- Manejo de errores con mensajes amigables al usuario

### Semana 4.5 — Reportes descargables
- Reporte ejecutivo en TXT con todas las secciones
- Reporte profesional en DOCX con tablas y formato Word
- Ambos incluyen: perfil, limpieza, correlaciones, modelo y limitaciones

---

## Instalacion local

### Requisitos
- Python 3.12+
- Git

### Pasos

```bash
# Clonar el repositorio
git clone https://github.com/Pepinip/Tabular-Insight-Workbench.git
cd Tabular-Insight-Workbench

# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows)
.venv\Scripts\activate

# Activar entorno (Mac/Linux)
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Correr la app
streamlit run app.py
```

### Correr los tests

```bash
python -m pytest tests/ -v
```

---

## Estructura del proyecto
Tabular-Insight-Workbench/
├── app.py                  # Entrypoint principal de Streamlit
├── requirements.txt        # Dependencias del proyecto
├── conftest.py             # Configuracion global de pytest
├── src/
│   ├── data_loader.py      # Carga y validacion de archivos
│   ├── profiler.py         # Perfilado de calidad del dataset
│   ├── cleaner.py          # Limpieza auditable
│   ├── analysis.py         # EDA: correlacion, histogramas, scatter
│   ├── modeling.py         # Regresion lineal, metricas, residuos
│   ├── reporting.py        # Generacion de reportes TXT y DOCX
│   └── logger.py           # Configuracion de logging
├── tests/
│   ├── test_data_loader.py
│   ├── test_profiler.py
│   ├── test_cleaner.py
│   ├── test_analysis.py
│   └── test_modeling.py
├── data/
│   └── sample/
│       └── sample_sales.csv
└── outputs/
├── figures/            # Graficas generadas
├── logs/               # Logs de sesion
└── reports/            # Reportes guardados
---

## Stack tecnico

| Herramienta | Version | Uso |
|---|---|---|
| Python | 3.12 | Lenguaje base |
| Streamlit | 1.58 | Framework web |
| Pandas | 3.x | Manipulacion de datos |
| NumPy | 2.x | Operaciones numericas |
| scikit-learn | 1.9 | Modelo de regresion |
| Matplotlib | 3.x | Visualizaciones |
| Seaborn | 0.13 | Heatmaps |
| python-docx | latest | Generacion de DOCX |
| pytest | 9.x | Testing |
---
## Dataset de prueba
El repositorio incluye `data/sample/sample_sales.csv` con:
- 8 filas (incluyendo 1 duplicada intencional)
- 1 valor nulo en `profit`
- Columna constante `country`
Ideal para probar todas las funciones de la app en segundos.
---
## Roadmap
| Version | Estado | Descripcion |
|---|---|---|
| v1.0 | ✅ Completa | Carga y perfilado |
| v2.0 | ✅ Completa | Limpieza auditable |
| v3.0 | ✅ Completa | EDA y modelo robusto |
| v4.0 | ✅ Completa | Tests, logging, errores |
| v4.5 | ✅ Completa | Reportes TXT y DOCX |
| v5.0 | 🔜 Planeada | Deteccion de outliers, variables categoricas |
| v6.0 | 🔜 Planeada | Modelos multiples (RF, Ridge, Lasso) |
| v7.0 | 🔜 Planeada | API REST con FastAPI |
---
## Contribuir
Este proyecto es open source bajo licencia GPL v3. Lee [CONTRIBUTING.md](CONTRIBUTING.md) para saber como contribuir.
---
## Licencia
[GNU General Public License v3.0](LICENSE)
Copyright (C) 2024 Pepinip
