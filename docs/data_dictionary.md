# Diccionario de datos — Student Alcohol Consumption

Dataset: UCI *Student Performance* (Cortez & Silva, 2008), curso 2005-2006,
dos escuelas portuguesas (GP y MS). **662 alumnos únicos** (1 fila por alumno,
priorizando Matemáticas; ver `docs/dataset_estructura.md`). El merge canónico del paper
(382 alumnos, `student-merge.R`) queda como referencia histórica.
Fuente Kaggle: `uciml/student-alcohol-consumption`.

## Target

| Nombre | Descripción | Tipo | Rango | Nulos | Disponibilidad | Riesgo de fuga | Tratamiento |
|---|---|---|---|---|---|---|---|
| `Walc` -> `walc_high` | Frecuencia de consumo de alcohol en **fin de semana** (auto-reporte). **Original ordinal 1-5** (1 = muy bajo, 5 = muy alto); el proyecto lo **binariza** por decisión operativa: alto = `Walc >= 3` | Binario (derivado de ordinal 1-5) | 0/1 | 0 | Fin de curso (encuesta) | Bajo | Target (la vista 1-5 se evalúa aparte en fase 17) |

## Features (29 originales)

| Nombre | Descripción | Tipo | Rango/Categorías | Nulos | Disponibilidad | Riesgo de fuga | Tratamiento |
|---|---|---|---|---|---|---|---|
| `school` | Escuela | Categórica | GP, MS | 0 | Inicio curso | No | OneHot |
| `sex` | Sexo | Binaria | F, M | 0 | Inicio curso | No (sensible -> auditoría equidad) | OneHot |
| `age` | Edad | Numérica | 15-22 | 0 | Inicio curso | No | Numérica (mediana) |
| `address` | Dirección | Binaria | U (urbana), R (rural) | 0 | Inicio curso | No | OneHot |
| `famsize` | Tamaño familiar | Binaria | LE3 (≤3), GT3 (>3) | 0 | Inicio curso | No | OneHot |
| `Pstatus` | Convivencia padres | Binaria | A (juntos), T (separados) | 0 | Inicio curso | No | OneHot |
| `Medu` | Educación madre | Ordinal | 0-4 | 0 | Inicio curso | No | Ordinal |
| `Fedu` | Educación padre | Ordinal | 0-4 | 0 | Inicio curso | No | Ordinal |
| `Mjob` | Trabajo madre | Categórica | teacher, health, services, at_home, other | 0 | Inicio curso | No | OneHot |
| `Fjob` | Trabajo padre | Categórica | teacher, health, services, at_home, other | 0 | Inicio curso | No | OneHot |
| `reason` | Razón de elección de escuela | Categórica | home, reputation, course, other | 0 | Inicio curso | No | OneHot |
| `guardian` | Tutor | Categórica | mother, father, other | 0 | Inicio curso | No | OneHot |
| `traveltime` | Tiempo de viaje | Ordinal | 1 (<15min) a 4 (>1h) | 0 | Inicio curso | No | Ordinal |
| `studytime` | Tiempo de estudio semanal | Ordinal | 1 (<2h) a 4 (>10h) | 0 | Inicio curso | No | Ordinal |
| `failures` | Asignaturas suspendidas | Ordinal | 0-3 | 0 | Inicio curso | No | Ordinal |
| `schoolsup` | Apoyo educativo extra | Binaria | yes/no | 0 | Inicio curso | No | OneHot |
| `famsup` | Apoyo familiar | Binaria | yes/no | 0 | Inicio curso | No | OneHot |
| `paid` | Clases extra pagadas | Binaria | yes/no | 0 | Inicio curso | No | OneHot |
| `activities` | Actividades extraescolares | Binaria | yes/no | 0 | Inicio curso | No | OneHot |
| `nursery` | Guardería | Binaria | yes/no | 0 | Inicio curso | No | OneHot |
| `higher` | Deseo de estudios superiores | Binaria | yes/no | 0 | Inicio curso | No | OneHot |
| `internet` | Internet en casa | Binaria | yes/no | 0 | Inicio curso | No | OneHot |
| `romantic` | Relación de pareja | Binaria | yes/no | 0 | Inicio curso | No | OneHot |
| `famrel` | Calidad relaciones familiares | Ordinal | 1 (muy mala) a 5 (excelente) | 0 | Inicio curso | No | Ordinal |
| `freetime` | Tiempo libre | Ordinal | 1 a 5 | 0 | Inicio curso | No | Ordinal |
| `goout` | Salir con amigos | Ordinal | 1 (nunca) a 5 (muy a menudo) | 0 | Inicio curso | No | Ordinal |
| `Dalc` | Consumo alcohol día laborable | Ordinal | 1 (muy bajo) a 5 (muy alto) | 0 | Inicio curso | No | Ordinal |
| `health` | Estado de salud percibido | Ordinal | 1 (muy malo) a 5 (muy bueno) | 0 | Inicio curso | No | Ordinal |
| `absences` | Faltas de asistencia | Numérica | 0-93 | 0 | Durante curso | No | Numérica (mediana) |

## Variables excluidas (post-evento / fuga)

| Nombre | Motivo |
|---|---|
| `G1`, `G2`, `G3` | Calificaciones de los periodos 1, 2 y final. Se conocen **después** del comportamiento que se predice y no están disponibles a inicio de curso -> eliminadas. Incluirlas dominaría el modelo y enmascararía los factores reales de consumo (fuga de información). |

## Features derivadas (feature engineering, fase 9)

| Nombre | Definición |
|---|---|
| `family_support` | `(schoolsup=yes) + (famsup=yes)` -> 0-2 |
| `study_intensity` | `studytime + (paid=yes)` -> 1-5 |
| `has_failures` | `failures > 0` -> 0/1 |
| `absences_high` | `absences >= 8` -> 0/1 |
| `parent_edu_max` | `max(Medu, Fedu)` |
| `parent_edu_diff` | `|Medu - Fedu|` |
| `social_exposure` | `goout + freetime` -> 2-10 |
| `health_low` | `health <= 2` -> 0/1 |

Total de features finales: **37** (29 originales + 8 derivadas).
