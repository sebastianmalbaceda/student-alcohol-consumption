"""
Carga, ingesta y auditoría de datos del proyecto Student Alcohol Consumption.

Fuente: UCI "Student Performance" (Cortez & Silva, 2008)
Kaggle: https://www.kaggle.com/datasets/uciml/student-alcohol-consumption
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas y configuración
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "archive"
DEFAULT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Columnas usadas para fusionar los dos datasets (según student-merge.R)
MERGE_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet",
]

# Variables post-evento: calificaciones (no disponibles en producción)
POST_EVENT_COLUMNS = ["G1", "G2", "G3"]

# Columnas que describen de forma única a cada alumno (id lógico)
ID_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime",
    "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
]

FEATURES_KEEP = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime",
    "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "health", "absences",
]


def load_student_dataset(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    mat_file: str = "student-mat.csv",
    por_file: str = "student-por.csv",
    sep: str = ",",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los dos datasets de matemáticas y portugués.

    Nota de auditoría: en esta copia de Kaggle el separador es coma
    (en el repositorio UCI original es punto y coma).
    """
    raw_dir = Path(raw_dir)
    mat = pd.read_csv(raw_dir / mat_file, sep=sep)
    por = pd.read_csv(raw_dir / por_file, sep=sep)
    merged = pd.merge(mat, por, on=MERGE_COLUMNS, suffixes=("_x", "_y"))
    return mat, por, merged


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva las features válidas (excluye G1, G2, G3 post-evento)."""
    return df[FEATURES_KEEP].copy()


def save_interim(df: pd.DataFrame, name: str, interim_dir: Path | str = DEFAULT_INTERIM_DIR) -> Path:
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_processed(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> Dict[str, Path]:
    """Guarda los conjuntos procesados y los índices de partición."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, obj in {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }.items():
        p = processed_dir / f"{name}.csv"
        obj.to_csv(p, index=False)
        paths[name] = p
    # Índices de las particiones (protocolo experimental, fase 7)
    indices = {
        "X_train": list(X_train.index),
        "X_val": list(X_val.index),
        "X_test": list(X_test.index),
    }
    (processed_dir / "split_indices.json").write_text(
        json.dumps(indices, indent=2), encoding="utf-8"
    )
    return paths


def load_processed(processed_dir: Path | str = DEFAULT_PROCESSED_DIR) -> Dict[str, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    out: Dict[str, pd.DataFrame] = {}
    for name in ["X_train", "X_val", "X_test"]:
        out[name] = pd.read_csv(processed_dir / f"{name}.csv")
    # Los targets se guardan como columna única -> devolverlos como Series
    for name in ["y_train", "y_val", "y_test"]:
        out[name] = pd.read_csv(processed_dir / f"{name}.csv").iloc[:, 0]
    return out


# ---------------------------------------------------------------------------
# Auditoría de datos (fase 3)
# ---------------------------------------------------------------------------

def audit_basic_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Devuelve un resumen de calidad básica del DataFrame."""
    report: Dict[str, Any] = {}
    report["n_rows"] = int(df.shape[0])
    report["n_cols"] = int(df.shape[1])
    report["n_missing_total"] = int(df.isna().sum().sum())
    report["n_missing_by_col"] = df.isna().sum().to_dict()
    report["n_duplicates_exact"] = int(df.duplicated().sum())
    report["n_constant_cols"] = int((df.nunique(dropna=False) <= 1).sum())
    report["constant_cols"] = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    # Tipos
    report["dtypes"] = {str(c): str(t) for c, t in df.dtypes.items()}
    # Valores imposibles (rangos esperados por dominio)
    report["out_of_range"] = {}
    for col in ["age"]:
        if col in df.columns:
            report["out_of_range"][col] = int(((df[col] < 10) | (df[col] > 22)).sum())
    for col in ["Medu", "Fedu", "traveltime", "studytime", "failures",
                "famrel", "freetime", "goout", "Dalc", "Walc", "health"]:
        if col in df.columns:
            bad = int(((df[col] < 0) | (df[col] > 5)).sum())
            report["out_of_range"][col] = bad
    return report


def _resolve(df: pd.DataFrame, col: str) -> str:
    """Devuelve el nombre real de la columna (soportando sufijos _x/_y del merge)."""
    if col in df.columns:
        return col
    if col + "_x" in df.columns:
        return col + "_x"
    return col


def audit_categories(df: pd.DataFrame) -> Dict[str, Any]:
    """Audita categorías inconsistentes en variables cualitativas."""
    inconsistent: Dict[str, list] = {}
    yes_no_cols = ["schoolsup", "famsup", "paid", "activities", "nursery",
                   "higher", "internet", "romantic"]
    for col in yes_no_cols:
        real = _resolve(df, col)
        if real not in df.columns:
            continue
        vals = set(df[real].dropna().unique())
        if not vals.issubset({"yes", "no"}):
            inconsistent[col] = sorted(vals)
    bin_cols = ["sex", "address", "famsize", "Pstatus"]
    for col in bin_cols:
        real = _resolve(df, col)
        if real not in df.columns:
            continue
        vals = set(df[real].dropna().unique())
        if len(vals) > 2:
            inconsistent[col] = sorted(vals)
    return {"inconsistent_categories": inconsistent}
