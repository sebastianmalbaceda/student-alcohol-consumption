"""
API REST de inferencia (fase 21) con FastAPI.

Ejecutar:  uvicorn src.api.main:app --reload
Docs:      http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import joblib
from typing import List, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.load_data import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(
    title="Student Alcohol Consumption API",
    version="1.0.0",
    description="Predice el riesgo de consumo alto de alcohol en fin de semana "
                "de estudiantes (escala 1-5, positivo si >= 3).",
)

# ---------------------------------------------------------------------------
# Esquema de entrada (fase 21.3: validación de esquema, tipos y rangos)
# ---------------------------------------------------------------------------

ORDINAL_RANGES = {
    "traveltime": (1, 4), "studytime": (1, 4), "failures": (0, 3),
    "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5),
    "Dalc": (1, 5), "health": (1, 5),
}
ORDINAL_COLS = list(ORDINAL_RANGES)
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
NUMERIC_COLS = ["age", "Medu", "Fedu", "absences"]
REQUIRED_COLS = ORDINAL_COLS + CATEGORICAL_COLS + NUMERIC_COLS


class StudentFeatures(BaseModel):
    school: Literal["GP", "MS"] = "GP"
    sex: Literal["F", "M"] = "M"
    age: int = Field(ge=10, le=22)
    address: Literal["U", "R"] = "U"
    famsize: Literal["LE3", "GT3"] = "GT3"
    Pstatus: Literal["A", "T"] = "A"
    Medu: int = Field(ge=0, le=4)
    Fedu: int = Field(ge=0, le=4)
    Mjob: Literal["teacher", "health", "services", "at_home", "other"] = "other"
    Fjob: Literal["teacher", "health", "services", "at_home", "other"] = "other"
    reason: Literal["home", "reputation", "course", "other"] = "course"
    guardian: Literal["mother", "father", "other"] = "mother"
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=3)
    schoolsup: Literal["yes", "no"] = "no"
    famsup: Literal["yes", "no"] = "no"
    paid: Literal["yes", "no"] = "no"
    activities: Literal["yes", "no"] = "no"
    nursery: Literal["yes", "no"] = "yes"
    higher: Literal["yes", "no"] = "yes"
    internet: Literal["yes", "no"] = "yes"
    romantic: Literal["yes", "no"] = "no"
    famrel: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0, le=93)


class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    label: str
    threshold: float
    features_used: int


# ---------------------------------------------------------------------------
# Carga del pipeline
# ---------------------------------------------------------------------------

_pipeline = None
_calibrator = None
_meta = None


def _load_metadata() -> dict:
    """Carga (una sola vez) los metadatos del modelo: features y umbral congelado."""
    global _meta
    if _meta is None:
        path = MODELS_DIR / "final_model_metadata.json"
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail="Metadatos del modelo no encontrados. Ejecuta primero el notebook "
                       "10_entrenamiento_final o scripts/run_pipeline.py --phase models",
            )
        import json
        _meta = json.loads(path.read_text(encoding="utf-8"))
    return _meta


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        path = MODELS_DIR / "final_model.joblib"
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail="Modelo no encontrado. Ejecuta primero el notebook "
                       "10_entrenamiento_final o scripts/run_pipeline.py --phase models",
            )
        _pipeline = joblib.load(path)
    return _pipeline


def get_calibrator():
    """Carga el calibrador (sigmoidal Platt) si existe. En producción se aplica SIEMPRE
    que esté disponible para no divergir del pipeline evaluado en la fase 17."""
    global _calibrator
    if _calibrator is None:
        path = MODELS_DIR / "final_calibrator.joblib"
        if path.exists():
            _calibrator = joblib.load(path)
    return _calibrator


def predict_proba_series(X: pd.DataFrame) -> np.ndarray:
    """Probabilidad de consumo alto aplicando el calibrador si existe."""
    raw = get_pipeline().predict_proba(X)[:, 1]
    cal = get_calibrator()
    if cal is not None:
        return cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    return raw


def get_threshold() -> float:
    """Umbral de decisión congelado en metadatos (fase 16.3)."""
    return float(_load_metadata().get("threshold", 0.5))


def _to_dataframe(features: StudentFeatures) -> pd.DataFrame:
    """Convierte el input validado en DataFrame con las columnas exactas
    que el pipeline espera. Reconstruye las features derivadas igual que
    src.features.build_features.add_domain_features (mismo orden que entrenamiento)."""
    data = {c: getattr(features, c) for c in REQUIRED_COLS}
    df = pd.DataFrame([data])
    # Feature engineering idéntico al de entrenamiento (fase 9)
    df["family_support"] = (
        (df["schoolsup"] == "yes").astype(int) + (df["famsup"] == "yes").astype(int)
    )
    df["study_intensity"] = df["studytime"] + df["paid"].map({"yes": 1, "no": 0})
    df["has_failures"] = (df["failures"] > 0).astype(int)
    df["absences_high"] = (df["absences"] >= 8).astype(int)
    df["parent_edu_max"] = df[["Medu", "Fedu"]].max(axis=1)
    df["parent_edu_diff"] = (df["Medu"] - df["Fedu"]).abs()
    df["social_exposure"] = df["goout"] + df["freetime"]
    df["health_low"] = (df["health"] <= 2).astype(int)
    # Verificación de coherencia con el modelo entrenado
    expected = _load_metadata().get("features", [])
    if expected and not set(expected).issubset(set(df.columns)):
        missing = [c for c in expected if c not in df.columns]
        raise HTTPException(
            status_code=500,
            detail=f"El modelo espera features que la API no puede construir: {missing}.",
        )
    return df[expected] if expected else df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures):
    """Predicción individual: probabilidad de consumo alto (Walc >= 3)."""
    X = _to_dataframe(features)
    proba = float(predict_proba_series(X)[0])
    threshold = get_threshold()
    pred = int(proba >= threshold)
    return PredictionResponse(
        probability=round(proba, 4),
        prediction=pred,
        label="alto" if pred == 1 else "bajo",
        threshold=threshold,
        features_used=len(REQUIRED_COLS),
    )


@app.post("/predict_batch")
def predict_batch(features_list: List[StudentFeatures]):
    """Predicción por lotes (máximo 100 registros). Usa el mismo umbral congelado
    que /predict para mantener coherencia entre endpoints."""
    if len(features_list) > 100:
        raise HTTPException(status_code=422, detail="Máximo 100 registros por lote")
    if len(features_list) == 0:
        raise HTTPException(status_code=422, detail="El lote no puede estar vacío")
    df = pd.DataFrame([{c: getattr(f, c) for c in REQUIRED_COLS} for f in features_list])
    proba = predict_proba_series(df)
    threshold = get_threshold()
    return {"probabilities": [round(p, 4) for p in proba],
            "predictions": [int(p >= threshold) for p in proba],
            "threshold": threshold}


@app.get("/features")
def list_features():
    """Lista de features esperadas por el modelo (fase 21.3)."""
    return {"required_columns": REQUIRED_COLS, "ordinal_ranges": ORDINAL_RANGES}
