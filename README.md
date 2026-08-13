# Student Alcohol Consumption

Proyecto de Machine Learning completo (fases 0-23 del guía metodológica
`docs/methodology-guide.md`) para predecir el **riesgo de consumo alto de
alcohol en fin de semana** en estudiantes de secundaria.

> **Regla central seguida**: definir -> auditar -> dividir -> aprender solo con train ->
> seleccionar con validación/CV -> comprobar una vez con test -> empaquetar -> monitorizar.
> El test nunca influye en preprocessing, selección de variables, hiperparámetros o modelo.

## Problema

- **Tarea**: clasificación binaria (tabular) sobre un target **originalmente ordinal
  multiclase**.
- **Target original**: `Walc` (frecuencia de consumo en fin de semana, escala ordinal
  **1-5**: 1 = muy bajo, 5 = muy alto). Distribución: 38 % nivel 1, 22 % nivel 2,
  19 % nivel 3, 14 % nivel 4, 7 % nivel 5.
- **Binarización (decisión operativa validada, fase 5)**: `alto = Walc >= 3` (niveles
  3, 4, 5) vs `bajo = Walc <= 2` (niveles 1, 2), ~39 % de positivos. La decisión se
  **valida formalmente** en el notebook 03 con dos análisis: (1) un modelo multiclase
  5x5 muestra accuracy exacta 0.47 pero ±1 nivel 0.82 y error ordinal medio 0.77 (los
  errores son entre niveles vecinos; confundir 1 con 5 es rarísimo), y (2) la
  comparación de cortes >=2 / >=3 / >=4 con el mismo modelo y CV muestra que **>=3 es
  el punto de equilibrio**: máxima correlación con el ordinal (0.884), desbalance
  mínimo (ratio 1.55) y mejor recall. El corte >=4 tendría mejor ROC-AUC (0.882) pero
  F1 0.645 y desbalance 3.87; el >=2 pierde selectividad (61.5 % positivos).
  Semánticamente, el nivel 3 ("medium") ya muestra perfil de riesgo en menores
  (goout 3.34, absences 5.7) y se clasifica como alto. La matriz de confusión del
  modelo de producción es 2x2; la multiclase 5x5 se muestra en los notebooks 03 y 11.
- **Unidad**: alumno. **Decisión**: priorizar intervenciones preventivas de los orientadores.
- **Costes**: FN = 2 (riesgo sanitario), FP = 1 (alerta innecesaria).

## Dataset y licencia

- UCI *Student Performance* (Cortez & Silva, 2008), curso 2005-2006, 2 escuelas portuguesas.
- Kaggle: [uciml/student-alcohol-consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption) — **CC BY 4.0** (atribución requerida).
- **Dos ficheros** (`student-mat.csv` y `student-por.csv`) porque el estudio cubre dos
  asignaturas (Matemáticas y Portugués). Este proyecto usa el **dataset de 662 alumnos
  únicos** (una fila por alumno, priorizando Matemáticas), en lugar del merge interno
  del paper (382 alumnos, solo quienes cursan ambas). Más datos, misma coherencia y
  misma prevalencia de consumo (~39 %). Detalle completo en `docs/dataset_estructura.md`.
- 0 nulos, 0 duplicados; las calificaciones `G1/G2/G3` son **post-evento** y se excluyen (fuga).

## Estructura

```
├── README.md                 # Este documento
├── .gitignore
├── requirements.txt
├── LICENSE
├── configs/                  # config.yaml, experiments.yaml, final_features.json, best_params_lgbm.json
├── data/
│   ├── raw/archive/          # student-mat.csv, student-por.csv (NUNCA se modifican)
│   ├── interim/              # (datos intermedios si se generan)
│   └── processed/            # X_train/X_val/X_test, y_*, split_indices.json
├── notebooks/                # 19 notebooks, uno por fase (00 a 18)
├── src/
│   ├── data/                 # load_data.py, make_dataset.py
│   ├── features/             # build_features.py
│   ├── models/               # train_model.py
│   ├── evaluation/           # metrics.py
│   └── api/                  # main.py (FastAPI)
├── models/                   # final_model.joblib + metadatos (gitignored)
├── reports/                  # JSON de métricas, experiments.csv
│   └── figures/              # gráficos exportados (SHAP, ROC, ...)
├── scripts/                  # run_pipeline.py, batch_inference.py
├── tests/                    # test_api_validation.py (pytest)
└── docs/                     # informe_tecnico.md, model_card.md, data_dictionary.md, API.md
```

## Instalación

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

### 1) Notebooks (recomendado, fase a fase)

```bash
jupyter lab
# abrir notebooks/00 ... 17 en orden
```

Cada notebook cubre una fase del guía metodológica (00: gobierno/contexto, 01: ingesta,
02: EDA, 03: target/métricas/split, 04: preprocessing, 05: selección, 06: baselines,
07: CV, 08: tuning, 09: selección de modelo, 10: entrenamiento final, 11: test final,
12: errores/explicabilidad, 13: robustez (incluye el diagnóstico visual: curva de
aprendizaje, ROC/PR y calibración), 14: ética, 15: despliegue, 16: documentación,
17: monitorización).

### 2) Pipeline de principio a fin

```bash
python scripts/run_pipeline.py            # data -> modelos -> tests
python scripts/run_pipeline.py --phase data
python scripts/run_pipeline.py --phase models
python scripts/run_pipeline.py --phase tests
```

### 3) API REST

```bash
uvicorn src.api.main:app --reload --port 8000
# docs: http://127.0.0.1:8000/docs
```

### 4) Inferencia batch

```bash
python scripts/batch_inference.py --input data/raw/archive/student-mat.csv --output predicciones.csv
```

### 5) Tests

```bash
pytest -v tests/
```

## Resultados (test bloqueado)

Dataset: **662 alumnos únicos** (1 fila por alumno, priorizando Matemáticas).

| Métrica | Valor real |
|---|---|
| ROC-AUC | **0.766** |
| PR-AUC | **0.741** |
| F1 (umbral coste 0.34, calibrado) | **0.702** |
| Accuracy | 0.72 |
| Brier / ECE | 0.187 / 0.129 |
| n test | 133 |

*(Generados por `reports/final_evaluation.json` y `reports/resumen_resultados.json`.)*

**CV (5 folds, train)**: RandomForest 0.812 · LogisticRegression 0.811 · LightGBM 0.800 ·
CatBoost 0.799 · XGBoost 0.796 · DecisionTree 0.759 · KNN 0.683
(`reports/experiments.csv`). Tras el tuning (fase 14) y la selección con Repeated CV
(fase 09), **RandomForest tuneado** ganó y es el modelo final, con
`class_weight="balanced"` y **calibración sigmoidal (Platt)** — elegida frente a la
isotónica porque con la muestra ampliada la isotónica sobreajusta (comprime las
probabilidades y empeora el ECE).

**Nota sobre el ROC-AUC**: bajó de 0.876 (382 alumnos) a 0.766 (662 alumnos) porque el
test ahora es más grande y realista (133 vs 77 registros) y porque el modelo ya no
"memoriza" perfiles repetidos del merge interno. El PR-AUC mejoró (0.741 vs 0.618 con
la calibración anterior): la priorización sigue siendo útil.

**Modelo final**: RandomForest tuneado (Optuna), 29 features originales + 8 derivadas
(las derivadas se documentan como candidatas; el pipeline interno usa las 29 originales),
`class_weight="balanced"`, calibrador sigmoidal (`models/final_calibrator.joblib`),
pipeline completo en `models/final_model.joblib`, umbral de coste 0.34 (sobre
probabilidades calibradas), zona de abstención [0.30, 0.60).

### Curva de aprendizaje (20 puntos) y diagnóstico visual

El notebook **13_robustez** (sección 19.4) genera la evidencia gráfica de un buen
entrenamiento — integrada en la fase de robustez según el orden del índice (figuras en
`reports/figures/13_*.png`):

| Indicador de la curva de aprendizaje (20 puntos) | Valor | Criterio | Cumple |
|---|---|---|---|
| ROC-AUC validation (100 % datos) | 0.838 | ≥ 0.75 | [OK] |
| Subida de la curva de validation | +0.208 | > 0 (asciende) | [OK] |
| Meseta alcanzada (últimos pasos) | -0.002 | ~0 | [OK] |
| Gap train-validation final | 0.156 | ~0.10-0.15 (RF) | [OK] |
| Gap dentro de banda ±2σ (sin overfitting) | 0.138-0.192 | sin tendencia | [OK] |

Con 396 filas de train (vs 228 antes) la curva es **notablemente más suave**: sube de
forma continua desde ~0.74 hasta 0.845 (máximo en n=268) y se estabiliza en ~0.84, con
la media móvil mostrando una progresión limpia sin el ruido brusco del dataset pequeño.

- **Curva ROC** en test: forma de bandera, AUC = 0.766 ≫ diagonal. [OK]
- **Curva PR**: PR-AUC = 0.741 ≫ prevalencia (0.39). [OK]
- **Calibración**: ECE 0.129 (sigmoidal; la isotónica empeoraba a 0.23 con la muestra amplia). [OK]
- **Diagnóstico**: patrón estándar de modelo que generaliza; **sin overfitting ni
  underfitting** (el gap ~0.16 es el esperado en RandomForest por el bootstrap y se
  mantiene estable a lo largo de la curva).

**Predictores más influyentes (SHAP)**: `goout`, `Dalc`, `age`, `romantic`,
`absences`, `studytime`, `famrel`.

## Limitaciones

- **Comparación 382 vs 662**: el modelo de 662 alumnos es el correcto (maximiza datos
  reales y evalúa con más fiabilidad), pero su AUC global (0.766) es menor que el del
  merge de 382 (0.914) porque incluye a los alumnos solo-Portugués — mayoritariamente de
  la escuela MS — que son intrínsecamente más difíciles (AUC 0.60). La comparación justa
  por CV da 0.824 vs 0.846: la diferencia real es ~0.02. Análisis completo en
  `docs/comparacion_382_vs_662.md`.
- Datos de 2005-2006 y solo 2 escuelas -> generalización limitada.
- Target auto-reportado (deseabilidad social).
- Correlacional, no causal; requiere supervisión humana.
- Muestra moderada (662 alumnos únicos; antes 382 del merge interno). La curva de
  aprendizaje con 396 filas de train ya es suave y estable (meseta ~0.84); con más
  cohortes podría resolverse aún mejor.
- `reports/learning_curve.json` contiene los 20 puntos de la curva de aprendizaje para
  análisis posterior.

## Referencias

- Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student
  Performance*. Proc. 5th FUture BUsiness TEChnology Conference.
- Kaggle: [Student Alcohol Consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption).
- Google Cloud Architecture Center: *Guidelines for developing high-quality ML solutions*.

## Licencia

Código: MIT (ver `LICENSE`). Datos: CC BY 4.0 (atribución a Cortez & Silva y Kaggle/UCI).
