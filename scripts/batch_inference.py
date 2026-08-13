"""
Inferencia batch desde CSV (fase 21.2).

Uso:
    python scripts/batch_inference.py --input nuevo_cohorte.csv --output predicciones.csv

El CSV de entrada debe contener las 29 columnas originales (sin G1/G2/G3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import add_domain_features  # noqa: E402

ORIGINAL_COLS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime",
    "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "health", "absences",
]


def main():
    parser = argparse.ArgumentParser(description="Inferencia batch con el modelo final")
    parser.add_argument("--input", required=True, help="CSV de entrada (features originales)")
    parser.add_argument("--output", default="predicciones.csv", help="CSV de salida")
    args = parser.parse_args()

    pipeline = joblib.load(PROJECT_ROOT / "models" / "final_model.joblib")
    meta = json.loads((PROJECT_ROOT / "models" / "final_model_metadata.json").read_text(encoding="utf-8"))
    # Calibrador sigmoidal (mismo que usa la API): evita divergencia train/producción
    calibrator_path = PROJECT_ROOT / "models" / "final_calibrator.joblib"
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None

    df = pd.read_csv(args.input)
    missing = [c for c in ORIGINAL_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Faltan columnas en el CSV: {missing}")

    X = add_domain_features(df[ORIGINAL_COLS])
    # El modelo se entrenó con las 29 columnas originales (las derivadas las
    # descarta el ColumnTransformer); seleccionamos el orden de metadatos.
    X = X[[c for c in meta["features"] if c in X.columns]]
    proba = pipeline.predict_proba(X)[:, 1]
    if calibrator is not None:
        proba = calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
    threshold = float(meta.get("threshold", 0.5))
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["prob_consumo_alto"] = proba.round(4)
    out["prediccion"] = pred
    out["etiqueta"] = out["prediccion"].map({1: "alto", 0: "bajo"})
    out["abstencion"] = ((proba >= 0.30) & (proba < 0.60)).astype(int)
    out["umbral_aplicado"] = threshold
    out.to_csv(args.output, index=False)
    print(f"Predicciones guardadas en {args.output} ({len(out)} registros)")
    print("Distribución de predicciones:", out["prediccion"].value_counts().to_dict())
    print("Umbral aplicado:", threshold)


if __name__ == "__main__":
    main()
