# Comparación 382 vs 662 alumnos — ¿cuál es el mejor modelo?

> **Pregunta**: ¿el modelo con 662 alumnos únicos es mejor que el de 382 (merge interno)?
> **Respuesta corta**: **depende de la métrica y del subgrupo**. El modelo 662 usa más
> datos reales y es más honesto, pero su AUC global es menor porque incluye una
> subpoblación (alumnos que solo cursan Portugués, mayoritariamente de la escuela MS)
> que es **intrínsecamente más difícil de predecir**.

## 1. Comparación con el mismo protocolo

Ambos modelos usan: RandomForest tuneado (Optuna) con `class_weight="balanced"`,
calibración sigmoidal (Platt), split estratificado con semilla 42 y umbral de coste
FP=1, FN=2 sobre validation.

| Métrica | Modelo 382 (merge) | Modelo 662 (únicos) | Lectura |
|---|---|---|---|
| N total / train / val / test | 382 / 228 / 77 / 77 | 662 / 396 / 133 / 133 | El 662 tiene ~73 % más datos |
| **ROC-AUC test** | **0.914** | 0.766 | El 382 parece mucho mejor… |
| IC 95 % bootstrap del AUC | [0.84, 0.98] | [0.67, 0.85] | …pero su IC es enorme (n=77) |
| PR-AUC test | 0.869 | 0.741 | Ídem |
| F1 (umbral coste) | 0.825 | 0.702 | Ídem |
| Brier / ECE | 0.131 / 0.168 | 0.187 / 0.129 | El 662 calibra mejor (ECE) |
| **CV 5-fold (train)** | **0.846 ± 0.017** | **0.824 ± 0.021** | La diferencia real es ~0.02 |

**Conclusión de la tabla**: cuando se compara con CV (mucho más estable que un único
test de 77 filas), la diferencia entre ambos modelos es **solo ~0.02 de ROC-AUC** a
favor del 382. El 0.914 del test de 382 es en gran parte **suerte del split pequeño**:
con n=77 el IC 95 % va de 0.84 a 0.98, una horquilla enorme.

## 2. ¿Por qué el AUC del 662 es menor? (causa raíz)

Descomponiendo el test de 133 del modelo 662:

| Subgrupo del test | n | AUC | Prevalencia |
|---|---|---|---|
| Alumnos en **ambas** asignaturas | 85 | **0.812** | ~0.43 |
| Alumnos **solo Portugués** | 48 | **0.604** | ~0.37 |
| Escuela GP | 91 | 0.790 | 0.43 |
| Escuela MS | 42 | 0.714 | 0.31 |

Los 271 alumnos que solo cursan Portugués (el 41 % de los "nuevos" datos) son
**demográficamente distintos**:

| Variable | Ambas asignaturas (366) | Solo Portugués (271) |
|---|---|---|
| Escuela MS | 12 % | **67 %** |
| Sexo femenino | 53 % | 67 % |
| Dirección rural | 23 % | **42 %** |
| Edad media | 16.7 | 17.0 |
| `absences` media | 5.8 | 3.7 |

Es decir: **el dataset de 662 no es "más de lo mismo"** — añade una subpoblación
(escuela MS, rural, femenina) con un patrón de consumo distinto y más difícil de
predecir. El AUC de 0.604 en ese subgrupo arrastra la media global.

## 3. ¿Es correcto usar 662? — Sí, con matices

**A favor del 662**:
- Usa **todos los alumnos reales** del estudio (662), no descarta al 41 %.
- El test es más grande (133 vs 77) -> las métricas son más fiables (IC más estrechos).
- La curva de aprendizaje es más suave y estable (meseta ~0.84 en validation).
- El ECE (calibración) mejora (0.129 vs 0.168).
- Es la decisión **metodológicamente más honesta**: el merge de 382 descartaba
  sistemáticamente a la escuela MS, ocultando que el modelo rinde peor ahí.

**En contra (matices)**:
- El AUC global baja (0.766 vs 0.914), lo que puede leerse como "peor modelo".
- La subpoblación solo-por es difícil; si el caso de uso real es solo la escuela GP,
  el 382 (o un 662 filtrado a GP) daría mejores números.

**Veredicto**: el modelo 662 es el **adecuado** porque maximiza los datos reales, evalúa
con más fiabilidad y expone una limitación real (escuela MS) que el 382 ocultaba. El
"mejor resultado" del 382 era en parte **sobreajuste al split pequeño**.

## 4. ¿Se puede mejorar sin perder generalización?

Sí, con estas vías ordenadas por impacto/riesgo:

| Mejora | Impacto esperado | Riesgo de overfitting |
|---|---|---|
| **Features específicas de escuela MS** (interacciones school×goout, school×Dalc) | Medio-alto en MS | Bajo (interacción con señal) |
| **Modelo por subgrupo** (GP y MS separados) o pesos por escuela | Alto en MS | Medio (menos datos por grupo) |
| **Más datos** (otras cohortes/años) | Alto (curva aún no saturada) | Ninguno |
| **Target encoding con CV** de categóricas de alta cardinalidad | Bajo (cardinalidad baja aquí) | Medio |
| **Ensemble** (RF + GBM promediados, o stacking) | Bajo-medio | Medio (requiere CV anidada) |
| **Recalibración por subgrupo** (calibradores separados GP/MS) | Bajo (mejora ECE) | Bajo |
| AutoML (AutoGluon) | Medio | Requiere validación cuidadosa |

**Recomendación principal**: el mayor salto vendría de **datos adicionales** (la curva
de aprendizaje sigue subiendo ligeramente con más n) y de **interacciones school×hábitos**
para ayudar a la escuela MS.

## 5. ¿Qué tan cerca de lo mejor posible estamos?

Estimación razonable (con 662 alumnos, 29 features y un problema de auto-reporte):

- **Límite superior empírico (oráculo)**: un modelo con las mismas features pero datos
  ilimitados alcanzaría probablemente **ROC-AUC 0.85-0.90** en la población completa
  (la CV del 382 en su subpoblación homogénea llega a 0.846; el techo por ruido de
  etiqueta y features limitadas está en ~0.90).
- **Dónde estamos**: CV 5-fold de **0.824** sobre los 662.
- **Brecha restante**: ~0.03-0.08 de ROC-AUC, atribuible sobre todo a (a) la escuela MS
  (shift poblacional) y (b) la ausencia de features más finas (contexto social, uso de
  tiempo real, etc.) que el cuestionario no recoge.
- **En la subpoblación GP** (la mejor modelada) estamos a ~0.05 del techo; **en MS** a
  ~0.15, porque apenas hay datos (12 % de la muestra) y son más heterogéneos.

**Conclusión final**: el modelo 662 es la decisión correcta y el estado actual está en
~0.82-0.83 de ROC-AUC CV, cerca de lo razonablemente alcanzable con estos datos y
features (~0.85-0.90). Las mejoras más prometedoras son **más datos** y **modelar por
subgrupo o con interacciones school×hábitos**; el resto (tuning fino, ensambles) aportaría
décimas con riesgo de sobreajuste.
