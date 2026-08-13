"""
Partición estratificada del dataset (fase 7).

El dataset no tiene repetición de alumnos (cada fila es un alumno distinto del
mismo estudio) y no hay orden temporal que respetar: se usa StratifiedShuffleSplit.
El test queda bloqueado: los índices se guardan en data/processed/split_indices.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Divide X, y en train / validation / test estratificados.

    train = 60 %, validation = 20 %, test = 20 % (bloqueado).
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(sss.split(X, y))
    X_tv, y_tv = X.iloc[train_val_idx], y.iloc[train_val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size / (1.0 - test_size), random_state=random_state
    )
    train_idx, val_idx = next(sss2.split(X_tv, y_tv))
    X_train, y_train = X_tv.iloc[train_idx], y_tv.iloc[train_idx]
    X_val, y_val = X_tv.iloc[val_idx], y_tv.iloc[val_idx]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def check_stratification(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Tabla comparativa de prevalencia del target por conjunto."""
    return pd.DataFrame({
        "train": y_train.value_counts(normalize=True).sort_index(),
        "val": y_val.value_counts(normalize=True).sort_index(),
        "test": y_test.value_counts(normalize=True).sort_index(),
    }).fillna(0.0).round(4)
