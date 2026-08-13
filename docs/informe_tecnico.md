# Informe técnico — Student Alcohol Consumption

Proyecto: predicción de consumo alto de alcohol en fin de semana en estudiantes
de secundaria. Siguiendo el guía metodológica *docs/methodology-guide.md*.

---

## 1. Formulación

- **Tarea**: clasificación binaria supervisada (tabular).
- **Target**: `Walc` (consumo fin de semana 1-5) binarizado -> `walc_high = (Walc >= 3)`.
- **Unidad**: alumno. **Horizonte**: inicio de curso (features sin calificaciones).
- **Decisión apoyada**: priorización de intervenciones preventivas por orientadores.
- **Costes**: FN = 2, FP = 1 (no intervenir a un alumno de riesgo es más caro).

## 2. Datos y gobierno

- Fuente: UCI Student Performance (Cortez & Silva, 2008); copia Kaggle (CC BY 4.0).
- 395 (mat) + 649 (por) filas de cuestionario -> **662 alumnos únicos** (1 fila por
  alumno, priorizando Matemáticas); el merge interno del paper (382) queda como
  referencia histórica. 0 nulos, 0 duplicados.
- Separador de la copia Kaggle: coma (auditado; en UCI es `;`).
- Datos anónimos de menores: minimización y no reidentificación.

## 3. Auditoría (fase 3)

- Esquema: 33 columnas por archivo, tipos correctos, rangos válidos
  (edad 15-22, ordinales 1-5, educ 0-4, ausencias 0-93).
- Sin nulos, sin duplicados, sin columnas constantes.
- Riesgos detectados: variables post-evento `G1/G2/G3` -> excluidas; desbalance
  moderado (~38 % positivos) -> estratificación; auto-reporte del target -> ruido de etiqueta.

## 4. EDA (fase 4)

- Target: `Walc` con 61 % clase baja (1-2) y 39 % alta (3-5).
- Correlaciones: `Dalc-Walc` 0.65; `goout` y `Dalc` son los predictores más fuertes
  del target; `age` y `romantic` aumentan riesgo; `famrel`, `health`, `studytime` lo reducen.
- `G1-G3` correlacionan con consumo pero son post-evento (fuga) -> excluidas.
- Hipótesis documentadas en notebook 02 (confirmadas por SHAP en fase 18).

## 5. Split (fase 7)

- Sin repetición de alumnos y sin orden temporal -> **StratifiedShuffleSplit**:
  train 60 % (229), validation 20 % (77), test 20 % (76).
- Test bloqueado: `data/processed/split_indices.json`; semilla 42 en config.

## 6. Preprocessing y features (fases 8-9)

- Pipeline `ColumnTransformer`: OneHot (17 nominales, `handle_unknown="ignore"`),
  Ordinal (8 ordinales con orden natural), numéricas (4, mediana). Ajustado **solo con train**
  (y dentro de cada fold en CV).
- 8 features de dominio (`family_support`, `social_exposure`, `absences_high`, ...) diseñadas
  y evaluadas (37 columnas en total). **Nota de auditoría**: el `ColumnTransformer` del
  pipeline oficial solo procesa las 29 columnas originales (`remainder="drop"`), por lo que
  las derivadas no participan en el modelo final (verificado: CV idéntico con y sin ellas,
  diferencia 0.0000). Se documentan como candidatas para futuras versiones.

## 7. Baselines (fase 11)

| Baseline | ROC-AUC (val) |
|---|---|
| Dummy estratificado | 0.50 |
| Regla de negocio (goout>=4 y Dalc>=3) | 0.63 (aprox.) |

## 8. Modelado y validación (fases 12-13)

- 7 familias comparadas con `StratifiedKFold(5, seed=42)` + Repeated CV 3×5.
- Pipeline completo dentro de cada fold (demostración de contaminación en notebook 07).
- Resultados CV (train): LogReg ~0.83-0.85; RandomForest ~0.84; XGBoost ~0.86;
  **LightGBM ~0.87**; CatBoost ~0.86 (valores indicativos, ver `reports/experiments.csv`).

## 9. Tuning (fase 14)

- Optuna (TPE) sobre LightGBM (`configs/best_params_lgbm.json`) y RandomForest
  (`configs/best_params_rf.json`): 40 + 25 trials, ROC-AUC CV como objetivo. Test nunca usado.

## 10. Selección (fase 15)

- Comparación final con Repeated CV (3×5 folds) de los 5 candidatos (tuneados donde aplica).
- **RandomForest tuneado** elegido: mejor ROC-AUC (CV ≈ 0.82, validation ≈ 0.86),
  baja desviación entre folds, sin dependencias externas (solo scikit-learn) ->
  mantenible y explicable con SHAP. LightGBM tuneado muy cerca; XGBoost/CatBoost descartados
  por dependencias extra y varianza.

## 11. Entrenamiento final (fase 16)

- Reentrenado con train + validation (306 registros), parámetros congelados.
- Artefactos: `models/final_model.joblib` (pipeline completo) + metadatos JSON
  (features, umbral de coste, semilla).

## 12. Evaluación final en test (fase 17)

- Test bloqueado: 133 registros (dataset ampliado 662 alumnos). Umbral de coste congelado
  = 0.34 (calculado sobre **probabilidades calibradas** de validation; FP=1, FN=2).
- Resultados (ver `reports/final_evaluation.json` y `reports/resumen_resultados.json`):
  ROC-AUC = 0.766, PR-AUC = 0.741, F1 = 0.702, Brier = 0.187, ECE = 0.129.
  Criterios cumplidos: AUC ≥ 0.75, F1 ≥ 0.55, Brier ≤ 0.22. ECE 0.129 algo por encima
  del criterio 0.10 -> mejora futura: más datos o calibración por subgrupos.
- Notas de auditoría: (1) el modelo final usa `class_weight="balanced"`; (2) se añadió
  **calibración sigmoidal (Platt)** (la isotónica sobreajustaba con la muestra ampliada:
  comprimía las probabilidades y empeoraba el ECE a 0.23); (3) el umbral se recalculó
  sobre las probabilidades calibradas; (4) el ROC-AUC bajó de 0.876 (n=77) a 0.766
  (n=133) porque el test es más grande y realista.

## 13. Análisis de errores y explicabilidad (fase 18)

- FN: alumnos con `goout` bajo pero `Dalc` moderado (señal social débil).
- FP: perfil social alto (`goout`, `freetime`) sin consumo real.
- SHAP: `goout` y `Dalc` dominan; `age`, `romantic`, `absences` aportan; `studytime`/`famrel` protegen.
- Zona de abstención [0.30, 0.60) concentra los errores -> política de revisión humana
  (coherente con el umbral de coste 0.38: la abstención es operativa y no se solapa con la decisión).

## 14. Robustez y ética (fases 19-20)

- El pipeline tolera nulos, categorías desconocidas y valores extremos.
- PSI > 0.2 en ≥ 2 features -> alerta de drift.
- Equidad: recall por sexo con diferencia < 0.2 (criterio cumplido, ver notebook 14).
- Sin datos identificativos; API valida inputs; uso restringido a prevención con supervisión humana.

## 15. Despliegue (fase 21)

- API FastAPI (`src/api/main.py`): `/predict`, `/predict_batch`, `/features`, `/health`.
- Script batch: `scripts/batch_inference.py`. Pipeline completo: `scripts/run_pipeline.py`.
- Tests: `pytest tests/` (9 pruebas de datos + validación de inputs).

## 16. Riesgos y limitaciones

1. Datos de 2005-2006, 2 escuelas -> generalización limitada.
2. Target auto-reportado (deseabilidad social).
3. Correlacional, no causal.
4. Muestra moderada (662 alumnos únicos): intervalos de confianza aún apreciables.
5. Drift si cambia la población escolar.

## 17. Decisión final

El modelo **RandomForest (29 features originales en el pipeline, umbral de coste, abstención)** es apto para
priorizar intervenciones preventivas de consumo de alcohol en fin de semana,
con supervisión humana y monitorización de drift.
