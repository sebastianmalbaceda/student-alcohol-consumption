# Estructura del dataset: ¿por qué hay dos archivos y un merge en R?

## 1. ¿Por qué hay dos datasets (`student-mat.csv` y `student-por.csv`)?

El estudio original (Cortez & Silva, 2008) recogió datos de **dos asignaturas
distintas** en las mismas dos escuelas portuguesas:

| Archivo | Asignatura | Filas |
|---|---|---|
| `student-mat.csv` | **Matemáticas** | 395 |
| `student-por.csv` | **Portugués** | 649 |

Cada fila es un **cuestionario de un alumno en una asignatura**: contiene las mismas 33
variables (sociodemográficas, familiares, hábitos y las calificaciones G1/G2/G3 **de esa
asignatura**). Por tanto:

- Un alumno que curse **ambas** asignaturas aparece **dos veces** (una por asignatura),
  con las mismas respuestas sociodemográficas pero con calificaciones distintas.
- Un alumno que curse **solo una** asignatura aparece una única vez.

## 2. ¿Qué es `student-merge.R`?

Es el **script original de los autores** que fusiona los dos archivos para construir el
dataset de análisis (el que usan en su paper). Hace un **merge (inner join) por las 13
columnas que identifican al alumno**:

```r
d1 = read.table("student-mat.csv", sep=";", header=TRUE)
d2 = read.table("student-por.csv", sep=";", header=TRUE)
d3 = merge(d1, d2, by=c("school","sex","age","address","famsize","Pstatus",
                        "Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(d3))  # 382 estudiantes
```

Es decir: **solo se conservan los alumnos que cursan AMBAS asignaturas** (porque el
merge exige que las 13 columnas coincidan en los dos archivos). El resultado del paper
es **382 alumnos** — exactamente lo que reproduce nuestro `src/features/build_features.py`.

## 3. ¿Por qué el resultado es tan pequeño (382)?

El cálculo completo:

```
395 (mat) + 649 (por) = 1044 filas de cuestionario
pero hay 662 alumnos únicos:
  - 366 cursan ambas asignaturas
  - 25 cursan solo Matemáticas
  - 271 cursan solo Portugués

Merge interno (inner join) = alumnos en AMBAS = 382 filas (366 + 16 duplicados de id)
```

**La pérdida es doble:**

1. **Alumnos de una sola asignatura** (25 + 271 = 296) se descartan: el merge interno
   solo conserva a quienes aparecen en los dos archivos.
2. **Duplicados de "id lógico"**: hay 4 pares en `mat` y 12 pares en `por` donde dos
   alumnos **distintos** comparten las 13 columnas de merge (mismo perfil: misma
   escuela, sexo, edad, educación de los padres, etc., pero distinto tutor y distintas
   calificaciones). pandas los cruza en todas las combinaciones, lo que añade 16 filas
   espurias al merge (382 = 366 reales + 16 combinaciones cruzadas).

Por eso el merge interno del paper produce **382 filas** (y con ~16 combinaciones
cruzadas que deberían depurarse). **Este proyecto ya no usa ese merge**: usa el dataset
de 662 alumnos únicos (sección siguiente), con lo que el train pasa a 396 filas y la
curva de aprendizaje se suaviza.

## 4. ¿Cuántos datos "reales" hay en total?

| Concepto | Nº |
|---|---|
| Filas de cuestionario (mat + por) | 1044 |
| **Alumnos únicos** (sin duplicar id lógico) | **662** |
| Alumnos en ambas asignaturas | 366 |
| Alumnos solo en Matemáticas | 25 |
| Alumnos solo en Portugués | 271 |
| Merge interno del paper (382 = 366 + 16 combinaciones) | 382 |

**Conclusión**: el proyecto usa ahora el dataset de **662 alumnos únicos** (una fila por
alumno, priorizando Matemáticas cuando existe — misma fuente del target que el merge
original). El train pasa de 228 a 396 filas y la curva de aprendizaje es notablemente
más suave (meseta ~0.84 frente al ruido de antes). El merge interno del paper (382)
queda documentado como referencia histórica y disponible vía `use_unique_students: false`
en `configs/config.yaml`.

## 5. Impacto en la curva de aprendizaje

Con el dataset ampliado (train = 396 filas), la curva de aprendizaje de 20 puntos
muestra:

- **Despegue suave** (n ≈ 30-150): sube de ~0.74 a ~0.82.
- **Meseta estable** (n > 150): fluctúa entre 0.82 y 0.845 sin tendencia; el máximo
  (0.845 en n=268) está dentro de la banda de ruido; la pendiente posterior es ~0
  (meseta real en ~0.84).

La forma ya **no es "rara"**: con más datos la curva se suaviza y sigue el patrón
estándar de un modelo que generaliza (validación creciente -> meseta; gap estable
~0.16, esperado en RandomForest por el bootstrap).
