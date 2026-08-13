# Changelog

Todas las versiones notables de este proyecto se documentan en este fichero.
El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.1.0/) y el
versionado semántico [SemVer](https://semver.org/lang/es/).

## [1.0.0] - 2026-08-12

### Añadido

- **Estructura completa del proyecto** de machine learning end-to-end siguiendo las
  fases 0-23 de la guía metodológica (`docs/methodology-guide.md`).
- **Dataset de 662 alumnos únicos** (una fila por alumno, priorizando Matemáticas),
  construido a partir de los dos ficheros crudos del estudio UCI *Student Performance*
  (Cortez & Silva, 2008). El merge interno del paper (382 alumnos) queda documentado
  como referencia histórica en `docs/dataset_estructura.md`.
- **18 notebooks ejecutables** (fases 00-17): gobierno y contexto, ingesta y auditoría,
  EDA, definición de target y métricas, partición, preprocessing, feature engineering,
  selección de variables, baselines, modelado, validación cruzada, tuning con Optuna,
  selección de modelo, entrenamiento final, evaluación en test, análisis de errores y
  explicabilidad (SHAP), robustez con diagnóstico visual (curva de aprendizaje con
  20 puntos, curvas ROC/PR, calibración), ética y equidad, despliegue, documentación y
  monitorización.
- **Código fuente modular** en `src/`:
  - `src/data/`: carga, auditoría y partición de datos.
  - `src/features/`: construcción del dataset y feature engineering.
  - `src/models/`: pipelines, CV y entrenamiento.
  - `src/evaluation/`: métricas, calibración, monitorización (PSI) y registro de versiones.
  - `src/api/`: API REST de inferencia con FastAPI.
- **Modelo final**: RandomForest tuneado con Optuna, `class_weight="balanced"`,
  calibración sigmoidal (Platt) y umbral de coste (FP=1, FN=2). Artefactos en `models/`
  (pipeline, calibrador, metadatos y registro de versiones).
- **Validación de la decisión binaria** (notebook 03): análisis del target ordinal
  original (Walc 1-5), modelo multiclase 5x5 (accuracy exacta 0.47, +/-1 nivel 0.82,
  errores ordinales) y comparación de cortes >=2 / >=3 / >=4 (el corte >=3 maximiza la
  correlación con el ordinal 0.884 y equilibra las clases con ratio 1.55).
- **Resultados en test bloqueado** (n=133): ROC-AUC 0.766, PR-AUC 0.741, F1 0.702,
  Brier 0.187, ECE 0.129. Curva de aprendizaje con meseta en ~0.84 (validation).
- **API REST** (`src/api/main.py`): endpoints `/predict`, `/predict_batch`, `/features`
  y `/health`, con validación de esquema (Pydantic), umbral congelado y calibrador
  aplicado en producción.
- **Scripts**: `scripts/run_pipeline.py` (pipeline de principio a fin),
  `scripts/batch_inference.py` (inferencia batch).
- **Pruebas unitarias** (19 tests con pytest): auditoría de datos, validación de inputs
  de la API, coherencia de artefactos y partición.
- **Documentación profesional**: `README.md`, `docs/informe_tecnico.md`,
  `docs/model_card.md`, `docs/data_dictionary.md`, `docs/API.md`,
  `docs/dataset_estructura.md`, `docs/comparacion_382_vs_662.md` y
  `docs/methodology-guide.md`.
- **Ficheros de repositorio profesional**: LICENSE (MIT), `.gitignore`,
  `requirements.txt`, `pyproject.toml`, `Makefile`, `Dockerfile`, `.dockerignore`,
  `.editorconfig`, `.pre-commit-config.yaml`, CI (` .github/workflows/ci.yml`),
  `CONTRIBUTING.md`, `SECURITY.md` y `CODE_OF_CONDUCT.md`.
- **Informes y figuras** en `reports/`: JSON de métricas, curva de aprendizaje (20
  puntos) y 18 figuras (distribuciones, CV, tuning, ROC/PR, calibración, SHAP, matrices
  de confusión binaria y multiclase, curva de aprendizaje).

### Cambiado

- La partición de datos usa 662 alumnos únicos (train 396 / validation 133 / test 133),
  frente a los 382 del merge interno del paper (train 228 / val 77 / test 77).
- La calibración del modelo final es **sigmoidal (Platt)** en lugar de isotónica: con
  la muestra ampliada, la isotónica sobreajustaba (comprimía las probabilidades y
  empeoraba el ECE); la sigmoidal reparte las probabilidades por todo el rango sin
  alterar el ROC-AUC.

### Documentado

- Comparación metodológica 382 vs 662 (`docs/comparacion_382_vs_662.md`): el modelo de
  662 alumnos es el adecuado (maximiza datos reales y evalúa con más fiabilidad); la
  diferencia real por CV es ~0.02 (0.824 vs 0.846); los alumnos solo-Portugués
  (mayoritariamente de la escuela MS) son intrínsecamente más difíciles de predecir
  (AUC 0.604).
