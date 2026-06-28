# Guia de Contribucion

Gracias por tu interes en contribuir a Tabular Insight Workbench.

## Antes de empezar

- Lee el [README](README.md) para entender el proyecto
- Revisa los [issues abiertos](https://github.com/Pepinip/Tabular-Insight-Workbench/issues) para ver que hay pendiente
- Si quieres proponer algo nuevo, abre un issue antes de codear
## Como contribuir
### 1. Fork y clone
```bash
git clone https://github.com/TU_USUARIO/Tabular-Insight-Workbench.git
cd Tabular-Insight-Workbench
```
### 2. Crear rama

```bash
git checkout -b feat/nombre-de-tu-feature
```

Prefijos recomendados:
- `feat/` para nuevas funcionalidades
- `fix/` para correcciones de bugs
- `test/` para agregar o mejorar tests
- `docs/` para documentacion

### 3. Configurar entorno

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```
### 4. Desarrollar y testear
Antes de hacer commit, asegurate de que los tests pasan:
```bash
python -m pytest tests/ -v
```
Los 39 tests deben pasar. Si agregas funcionalidad, agrega los tests correspondientes.
### 5. Pull Request

- Haz commit con mensajes descriptivos en formato convencional:
- Abre un Pull Request hacia `main`
- Describe que cambiaste y por que

## Reglas de arquitectura

Estas reglas NO deben romperse:

- `raw_df` nunca se modifica directamente
- Toda limpieza genera `clean_df` como copia
- Toda limpieza tiene auditoria de impacto
- No hacer limpieza automatica silenciosa
- No inferir causalidad desde correlacion
- Mantener modulos separados por responsabilidad
- Todo codigo nuevo debe tener tests
## Estructura de modulos

| Modulo | Responsabilidad |
|---|---|
| `data_loader.py` | Carga y validacion de archivos |
| `profiler.py` | Perfilado no destructivo |
| `cleaner.py` | Limpieza auditable |
| `analysis.py` | EDA y visualizaciones |
| `modeling.py` | Modelo y metricas |
| `reporting.py` | Generacion de reportes |
| `logger.py` | Logging de sesiones |

## Preguntas
Abre un issue con la etiqueta `question` o escribe directamente en las discusiones del repo.