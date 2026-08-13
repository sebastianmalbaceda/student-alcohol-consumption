# Model Card — Student Alcohol Consumption

## Resumen

Modelo de clasificación binaria que estima la probabilidad de **consumo alto de
alcohol en fin de semana** (`Walc >= 3`) en estudiantes de secundaria, a partir de
29 variables sociodemográficas, familiares y académicas (37 con feature engineering).

- **Modelo**: RandomForest (ensemble de árboles) tuneado con Optuna,
  `class_weight=balanced`.
- **Umbral de decisión**: óptimo por coste (FP=1, FN=2) ajustado en validation.
- **Métrica primaria**: ROC-AUC.

## Uso previsto

- Priorización de programas de prevención y seguimiento por orientadores escolares.
- Batch anual por cohorte; inferencia por API o CSV.
- Decisiones **siempre con supervisión humana** (el modelo no diagnostica).

## Uso no previsto

- Diagnóstico clínico de alcoholismo o trastornos por consumo.
- Evaluación individual definitiva, sanciones o decisiones académicas.
- Aplicación a poblaciones fuera de Portugal / de otra época sin revalidación.
- Inferencia causal sobre los factores de consumo.

## Datos

- UCI *Student Performance* (Cortez & Silva, 2008), 2 escuelas portuguesas, curso 2005-2006.
- **662 alumnos únicos** (1 fila por alumno, priorizando Matemáticas); el merge interno
  del paper (382) queda como referencia histórica.
- 0 nulos, 0 duplicados; 29 features + 8 derivadas.
- **Target original `Walc` ordinal multiclase (1-5)**; el modelo de producción es binario
  (alto = `Walc >= 3`). En la fase 17 se incluye además una matriz multiclase 5x5.
- **Excluidas** las calificaciones G1/G2/G3 (post-evento, fuga).
- Licencia: CC BY 4.0 (Kaggle). Anónimo.

## Métricas (test bloqueado)

| Métrica | Valor |
|---|---|
| ROC-AUC | **0.766** |
| PR-AUC | 0.741 |
| F1 (umbral coste 0.34) | 0.702 |
| Brier / ECE | 0.187 / 0.129 |
| n test | 133 |

*(Valores exactos en `reports/final_evaluation.json` y `reports/resumen_resultados.json`.)*

El modelo final incluye **calibración sigmoidal (Platt)** ajustada sobre validation: con
la muestra ampliada, la isotónica sobreajustaba (comprimía las probabilidades y empeoraba
el ECE); la sigmoidal reparte las probabilidades por todo el rango sin alterar el ROC-AUC.

## Subgrupos (equidad)

- Recalls por sexo y escuela con diferencia ≤ 0.2 (criterio de aceptación).
- El dataset tiene desbalance de escuela (GP mayoritaria) -> métricas por subgrupo en notebook 14.

## Limitaciones

- Muestra pequeña y de 2005-2006 (2 escuelas) -> generalización limitada.
- Target auto-reportado (sesgo de deseabilidad social).
- Correlacional, no causal.
- Los errores se concentran en la zona de probabilidad 0.30-0.60 -> abstención recomendada.

## Consideraciones éticas

- Población menor de edad: minimización de datos, sin identificadores.
- Las variables `sex`, `age`, `romantic` pueden actuar como proxies; se auditan
  por subgrupo y su peso es público (SHAP).
- El modelo refuerza la capacidad de los orientadores; no sustituye el juicio humano.

## Mantenimiento y versiones

- **v1.0.0** (2026-08-12): RandomForest tuneado con Optuna, calibración
  sigmoidal. Estado: producción.
- Criterios de reentrenamiento: PSI > 0.2 en ≥ 2 features; AUC < 0.70 en nuevos datos;
  o ≥ 500 registros nuevos.
- Proceso: auditoría -> nuevo test bloqueado -> comparación -> aprobación -> versionado
  (`models/registry.json`) -> rollback automático.
- Registro de cambios: `docs/CHANGELOG.md`.

## Contacto / autoría

**Autor**: Sebastián Malbaceda Leyva

Proyecto académico. Repositorio: raíz del proyecto (README.md).
