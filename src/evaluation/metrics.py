"""
Evaluación de modelos: métricas, calibración, umbral óptimo, fairness y
análisis de errores (fases 6, 17, 18, 20).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

from src.data.load_data import PROJECT_ROOT

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# Costes de la matriz de decisión (fase 6.3): FN más grave que FP
COST_MATRIX = {"tn": 0.0, "fp": 1.0, "fn": 2.0, "tp": 0.0}


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Métricas primarias y secundarias (fase 6.1/6.2)."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cost = (COST_MATRIX["fp"] * fp + COST_MATRIX["fn"] * fn) / len(y_true)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(_pr_auc(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "threshold": float(threshold),
        "cost": float(cost),
        "n_tp": int(tp), "n_fp": int(fp), "n_fn": int(fn), "n_tn": int(tn),
    }


def _pr_auc(y_true, y_proba) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_proba))


def find_optimal_threshold(
    y_true: pd.Series | np.ndarray, y_proba: np.ndarray,
    cost_matrix: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Umbral óptimo por coste de decisión (fase 6.3)."""
    cm = cost_matrix or COST_MATRIX
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (cm["fp"] * fp + cm["fn"] * fn) / len(y_true)
        if cost < best_cost:
            best_cost, best_t = cost, t
    return {"optimal_threshold": float(best_t), "min_cost": float(best_cost)}


def calibration_summary(
    y_true: pd.Series | np.ndarray, y_proba: np.ndarray, n_bins: int = 10,
) -> Dict[str, float]:
    """Brier, ECE (Expected Calibration Error) y curva de calibración."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    return {
        "brier": float(brier_score_loss(y_true, y_proba)),
        "ece": ece,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


def roc_curve_data(y_true, y_proba) -> Dict[str, list]:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def pr_curve_data(y_true, y_proba) -> Dict[str, list]:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {"precision": precision.tolist(), "recall": recall.tolist()}


def learning_curve_data(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    train_sizes: np.ndarray | list | None = None,
    n_points: int = 20,
    scoring: str = "roc_auc",
    n_jobs: int = 1,
    random_state: int = 42,
) -> Dict:
    """Curva de aprendizaje con N puntos (por defecto 20) sobre train y validation.

    Devuelve los tamaños de entrenamiento y las medias/desviaciones de la métrica
    (por defecto ROC-AUC) para train y validation a lo largo de la curva.
    Un buen modelo debe mostrar: (1) train_score decreciente y estable,
    (2) val_score creciente que se aplana, y (3) convergencia entre ambas sin
    una brecha grande (ni overfitting ni underfitting).

    Parámetros
    ----------
    estimator : estimador sklearn (normalmente un Pipeline)
    X, y : datos de entrenamiento
    cv : objeto cross-validator (StratifiedKFold recomendado)
    train_sizes : fracciones del train (si None, np.linspace(0.05, 1.0, n_points))
    n_points : número de puntos de la curva (por defecto 20)
    scoring : métrica (por defecto "roc_auc")
    n_jobs : paralelización (1 por defecto para evitar memmapping en Windows)
    """
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.05, 1.0, n_points)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes,
        n_jobs=n_jobs, random_state=random_state,
    )
    return {
        "train_sizes": [int(s) for s in train_sizes],
        "train_scores_mean": [float(v) for v in train_scores.mean(axis=1)],
        "train_scores_std": [float(v) for v in train_scores.std(axis=1)],
        "val_scores_mean": [float(v) for v in val_scores.mean(axis=1)],
        "val_scores_std": [float(v) for v in val_scores.std(axis=1)],
    }


def fairness_by_group(
    y_true: pd.Series, y_proba: np.ndarray, groups: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Métricas por subgrupo (fase 20.1): accuracy, F1, tasa de positivos,
    recall y diferencia (equalized odds simplificado)."""
    rows = []
    y_pred = (y_proba >= threshold).astype(int)
    for g in groups.unique():
        mask = groups.values == g
        yt, yp, proba = y_true.values[mask], y_pred[mask], y_proba[mask]
        rows.append({
            "group": str(g),
            "n": int(mask.sum()),
            "prevalence": float(yt.mean()),
            "predicted_positive_rate": float(yp.mean()),
            "accuracy": float(accuracy_score(yt, yp)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "roc_auc": float(roc_auc_score(yt, proba)) if len(np.unique(yt)) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def error_analysis(
    y_true: pd.Series, y_proba: np.ndarray, X: pd.DataFrame,
    threshold: float = 0.5, top_k: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Análisis de errores (fase 18.1): FP, FN y casos de alta confianza erróneos."""
    y_pred = (y_proba >= threshold).astype(int)
    df = X.copy()
    df["y_true"] = y_true.values
    df["y_proba"] = y_proba
    df["y_pred"] = y_pred
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)
    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].sort_values("y_proba", ascending=False)
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].sort_values("y_proba", ascending=True)
    return {
        "false_positives": fp.head(top_k),
        "false_negatives": fn.head(top_k),
        "n_fp": int(len(fp)),
        "n_fn": int(len(fn)),
    }


def save_evaluation_report(
    metrics: Dict[str, float],
    fairness: pd.DataFrame,
    error_summary: Dict[str, int],
    path: Path | str,
) -> None:
    """Guarda el informe de evaluación final (fase 17/18)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics": metrics,
        "error_summary": error_summary,
        "fairness": fairness.to_dict(orient="records"),
    }
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def load_final_pipeline(models_dir: Path | str = DEFAULT_MODELS_DIR):
    """Carga el pipeline final entrenado (para API y evaluación)."""
    return joblib.load(Path(models_dir) / "final_model.joblib")


def multiclass_evaluation(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
) -> Dict:
    """Evaluación multiclase (fase 17.5): accuracy, matriz de confusión y errores
    por distancia ordinal.

    Para un target ordinal (como Walc 1-5) es clave medir NO solo el acierto exacto,
    sino la DISTANCIA del error: confundir 3 con 4 (error de 1 nivel) es mucho menos
    grave que confundir 1 con 5 (error de 4 niveles).

    Devuelve:
    - accuracy: proporción de aciertos exactos
    - accuracy_1off: proporción de aciertos dentro de +/-1 nivel
    - matriz: matriz de confusión (list de list)
    - error_medio: distancia ordinal media (0 = perfecto, 4 = máximo)
    - errores_por_distancia: {distancia: nº de errores}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))

    # Distancia ordinal por pares
    dist = np.abs(y_true.astype(int) - y_pred.astype(int))
    acc_1off = float((dist <= 1).mean())
    error_medio = float(dist.mean())
    from collections import Counter
    errores_por_distancia = dict(sorted(Counter(dist.tolist()).items()))

    return {
        "labels": labels,
        "accuracy": acc,
        "accuracy_1off": acc_1off,
        "error_medio": error_medio,
        "matriz": cm.tolist(),
        "errores_por_distancia": errores_por_distancia,
    }
