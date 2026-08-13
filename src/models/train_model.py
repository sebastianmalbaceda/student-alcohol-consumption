"""
Entrenamiento de modelos con pipelines seguros (fase 11-16).

Todo preprocessing (OneHotEncoder / OrdinalEncoder / imputación) se ajusta
dentro de cada fold de CV y dentro de train; nunca con test.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data.load_data import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Columnas
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

ORDINAL_COLS = [
    "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
    "Dalc", "health",
]

ORDINAL_ORDERS = {
    "traveltime": [1, 2, 3, 4],
    "studytime": [1, 2, 3, 4],
    "failures": [0, 1, 2, 3],
    "famrel": [1, 2, 3, 4, 5],
    "freetime": [1, 2, 3, 4, 5],
    "goout": [1, 2, 3, 4, 5],
    "Dalc": [1, 2, 3, 4, 5],
    "health": [1, 2, 3, 4, 5],
}

NUMERIC_COLS = ["age", "Medu", "Fedu", "absences"]

# ---------------------------------------------------------------------------
# Preprocessor y pipelines
# ---------------------------------------------------------------------------

def get_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Preprocessor seguro: se ajusta solo con train (o el train de cada fold)."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        *((("scaler", StandardScaler()),) if scale_numeric else ()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[ORDINAL_ORDERS[c] for c in ORDINAL_COLS],
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
        ("ord", ordinal_pipe, ORDINAL_COLS),
    ], remainder="drop")


def make_pipeline(model: Any, scale_numeric: bool = False) -> Pipeline:
    return Pipeline([
        ("preprocessor", get_preprocessor(scale_numeric=scale_numeric)),
        ("model", model),
    ])


def get_model_factories(seed: int = 42) -> Dict[str, Any]:
    """Fábricas de los modelos candidatos (fase 12: secuencia de complejidad)."""
    return {
        "BaselineDummy": lambda: DummyClassifier(strategy="stratified", random_state=seed),
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=2000, C=1.0, random_state=seed),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=15),
        "DecisionTree": lambda: DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=seed),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=seed, n_jobs=-1),
        "XGBoost": lambda: _xgboost(seed),
        "LightGBM": lambda: _lightgbm(seed),
        "CatBoost": lambda: _catboost(seed),
    }


def _xgboost(seed: int):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.0,
        eval_metric="auc", random_state=seed, n_jobs=-1,
        early_stopping_rounds=None, verbosity=0)


def _lightgbm(seed: int):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbose=-1)


def _catboost(seed: int):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.05,
        auto_class_weights="Balanced", random_seed=seed,
        verbose=False, allow_writing_files=False)


def select_features_available(
    X: pd.DataFrame, post_event: List[str] = None
) -> pd.DataFrame:
    """Elimina variables no disponibles en producción (post-evento)."""
    drop = set(post_event or [])
    return X.drop(columns=[c for c in drop if c in X.columns])


# ---------------------------------------------------------------------------
# CV y entrenamiento
# ---------------------------------------------------------------------------

def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_factory: Any,
    cv: StratifiedKFold,
    metric: str = "roc_auc",
    scale_numeric: bool = False,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """CV con pipeline dentro de cada fold (fase 13).

    n_jobs=1 por defecto: evita problemas de memmapping de joblib en Windows
    dentro de kernels embebidos; el dataset es pequeño y la diferencia es mínima.
    """
    t0 = time.time()
    scores = cross_val_score(
        make_pipeline(model_factory(), scale_numeric=scale_numeric),
        X, y, cv=cv, scoring=metric, n_jobs=n_jobs,
    )
    return {
        "model": model_name,
        "metric": metric,
        "scores": scores.tolist(),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "time_s": round(time.time() - t0, 2),
    }


def train_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "LightGBM",
    use_val: bool = True,
    random_state: int = 42,
    scale_numeric: bool = False,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    calibrate: bool = True,
) -> Dict[str, Any]:
    """Entrena el modelo final con train (+ validation) y guarda artefactos.

    Si calibrate=True, ajusta un calibrador sigmoidal (Platt) sobre validation
    (fase 18.3) y lo guarda como paso adicional del pipeline: mejora la calibración
    (ECE) sin alterar el ranking (ROC-AUC) del modelo. Se usa sigmoidal en lugar de
    isotónico porque con la muestra ampliada la isotónica sobreajusta.
    """
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    factories = get_model_factories(seed=random_state)
    model = factories[model_name]()

    # El target de train + val (la selección ya terminó: fase 16)
    if use_val:
        X_fit = pd.concat([X_train, X_val], axis=0)
        y_fit = pd.concat([y_train, y_val], axis=0)
    else:
        X_fit, y_fit = X_train, y_train

    pipeline = make_pipeline(model, scale_numeric=scale_numeric)
    t0 = time.time()
    pipeline.fit(X_fit, y_fit)
    fit_time = round(time.time() - t0, 2)

    # Recalibración sigmoidal (Platt) sobre validation (solo si hay validation y calibrate)
    calibrator = None
    if calibrate and use_val:
        from sklearn.linear_model import LogisticRegression
        p_val = pipeline.predict_proba(X_val)[:, 1]
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(p_val.reshape(-1, 1), y_val)

    def _predict_proba(X):
        p = pipeline.predict_proba(X)[:, 1]
        if calibrator is not None:
            return calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
        return p

    # Probabilidades sobre test (solo para reportar; el test está bloqueado)
    y_proba = _predict_proba(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    test_auc = float(roc_auc_score(y_test, y_proba))

    artifacts = {
        "model_name": model_name,
        "random_state": random_state,
        "use_val": use_val,
        "calibrated": calibrate and use_val,
        "n_train": int(len(X_fit)),
        "fit_time_s": fit_time,
        "test_roc_auc": test_auc,
        "features": list(X_fit.columns),
    }

    # Guardar artefactos (fase 16.3): pipeline + calibrador
    joblib.dump(pipeline, models_dir / "final_model.joblib")
    if calibrator is not None:
        joblib.dump(calibrator, models_dir / "final_calibrator.joblib")
    (models_dir / "final_model_metadata.json").write_text(
        json.dumps(artifacts, indent=2), encoding="utf-8")

    # Predicciones de test guardadas para la evaluación bloqueada
    pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }).to_csv(reports_dir / "test_predictions.csv", index=False)

    logger.info("Modelo final %s entrenado. ROC-AUC test (solo informativo): %.4f",
                model_name, test_auc)
    return artifacts
