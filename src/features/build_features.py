"""
Feature engineering (fase 9): construcción del dataset con el merge x/y
(sufijos _x y _y) y creación de features de conocimiento de dominio.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Variables originales conservadas (ambos datasets x/y)
FEATURES_KEEP = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime",
    "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "health", "absences",
]

# Variables post-evento que NUNCA entran como features
POST_EVENT = ["G1", "G2", "G3"]

ABSENCES_HIGH_THRESHOLD = 8


def build_dataset(
    mat: pd.DataFrame,
    por: pd.DataFrame,
    merge_columns: List[str],
) -> pd.DataFrame:
    """Fusiona ambos datasets (como student-merge.R) y devuelve features + target.

    Tras el merge, las columnas no usadas como clave llevan sufijos _x/_y;
    se resuelven tomando siempre la copia _x (equivalente a la _y: mismas
    respuestas del mismo alumno en ambos cuestionarios).
    """
    merged = pd.merge(mat, por, on=merge_columns, suffixes=("_x", "_y"))
    df = pd.DataFrame(index=merged.index)
    for col in FEATURES_KEEP:
        if col in merged.columns:
            df[col] = merged[col]
        else:
            df[col] = merged[col + "_x"]
    # Target: consumo de fin de semana del cuestionario de matemáticas
    df["Walc"] = merged["Walc_x"]
    return df


def build_dataset_unique_students(
    mat: pd.DataFrame,
    por: pd.DataFrame,
    merge_columns: List[str],
) -> pd.DataFrame:
    """Construye el dataset con TODOS los alumnos únicos (662) en lugar del merge
    interno (382).

    Motivación (ver docs/dataset_estructura.md):
    - Los ficheros crudos contienen 1044 filas de cuestionario (395 mat + 649 por)
      porque cada alumno aparece una vez por asignatura cursada.
    - El merge interno del paper conserva solo a los 382 alumnos que cursan AMBAS
      asignaturas, descartando a 280 alumnos reales (25 solo mat + 271 solo por).
    - Este dataset construye **una fila por alumno único**: si el alumno está en
      Matemáticas se usa su fila de Matemáticas (misma fuente del target que el
      proyecto actual); si solo está en Portugués se usa su fila de Portugués.
    - Las variables sociodemográficas y de hábitos coinciden ~97-99% entre
      asignaturas para el mismo alumno (verificado), por lo que la fusión es
      coherente. Las variables específicas de asignatura (paid, absences) se
      toman de la asignatura elegida.

    El id lógico del alumno son las 13 columnas de merge (igual que student-merge.R).
    Limitación documentada: 4 pares en mat y 12 en por comparten el mismo id siendo
    alumnos distintos (mismo perfil, distinto tutor/notas); se conserva el primero.
    """
    mat = mat.copy()
    por = por.copy()
    mat["_student_id"] = mat[merge_columns].astype(str).agg("|".join, axis=1)
    por["_student_id"] = por[merge_columns].astype(str).agg("|".join, axis=1)

    # 1 fila por alumno en mat (prioridad: fuente del target actual)
    df_mat = mat.drop_duplicates("_student_id", keep="first").copy()
    ids_mat = set(df_mat["_student_id"])

    # Alumnos que solo están en por
    df_por_solo = por[~por["_student_id"].isin(ids_mat)].drop_duplicates(
        "_student_id", keep="first").copy()

    df = pd.concat([df_mat, df_por_solo], axis=0, ignore_index=True)
    df = df.drop(columns=["_student_id"])
    # Mismo orden de columnas que el dataset estándar
    df = df[FEATURES_KEEP + ["Walc"]]
    return df


def add_domain_features(df: pd.DataFrame, absences_high_threshold: int = ABSENCES_HIGH_THRESHOLD) -> pd.DataFrame:
    """Añade features de conocimiento de dominio (solo derivadas de features
    disponibles en producción, sin tocar el target)."""
    out = df.copy()
    # Interacción apoyo familiar (padres que ayudan en estudios o en casa)
    out["family_support"] = (
        (out.get("schoolsup", "no") == "yes").astype(int)
        + (out.get("famsup", "no") == "yes").astype(int)
    )
    # Carga académica: tiempo de estudio + clases extra pagadas
    out["study_intensity"] = out["studytime"] + out.get("paid", "no").map({"yes": 1, "no": 0})
    # Riesgo académico: asignaturas suspendidas (indicador)
    out["has_failures"] = (out["failures"] > 0).astype(int)
    # Absentismo alto (binario)
    out["absences_high"] = (out["absences"] >= absences_high_threshold).astype(int)
    # Proporción de padres con estudios superiores (0-4 escala educativa)
    out["parent_edu_max"] = out[["Medu", "Fedu"]].max(axis=1)
    out["parent_edu_diff"] = (out["Medu"] - out["Fedu"]).abs()
    # Vida social: salir con amigos + tiempo libre (proxy de exposición social)
    out["social_exposure"] = out["goout"] + out["freetime"]
    # Salud auto-percibida como factor protector
    out["health_low"] = (out["health"] <= 2).astype(int)
    return out


FEATURES_ENGINEERED = FEATURES_KEEP + [
    "family_support", "study_intensity", "has_failures", "absences_high",
    "parent_edu_max", "parent_edu_diff", "social_exposure", "health_low",
]

FINAL_FEATURE_LIST = FEATURES_ENGINEERED
