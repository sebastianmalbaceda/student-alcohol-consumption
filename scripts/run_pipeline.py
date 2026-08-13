"""
Script ejecutable de principio a fin (fase 1.2: reproducibilidad).

Uso:
    python scripts/run_pipeline.py            # todo el pipeline
    python scripts/run_pipeline.py --phase data|eda|models|api|tests
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def phase_data():
    """Fases 0-10: carga, auditoría, EDA, split, preprocessing, features."""
    import yaml

    from src.data.load_data import (
        audit_basic_quality, load_student_dataset,
        select_features,
    )
    from src.data.make_dataset import check_stratification, stratified_split
    from src.features.build_features import (
        add_domain_features, build_dataset, build_dataset_unique_students,
    )
    from src.data.load_data import save_processed

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    merge_cols = cfg["data"]["merge_columns"]
    use_unique = cfg["data"].get("use_unique_students", False)

    mat, por, merged = load_student_dataset()
    if use_unique:
        df = build_dataset_unique_students(mat, por, merge_cols)
        print("[data] Dataset de alumnos ÚNICOS (662 filas, 1 por alumno)")
    else:
        df = build_dataset(mat, por, merge_cols)
        print("[data] Dataset del merge interno del paper (382 filas)")
    df = add_domain_features(df)
    audit = audit_basic_quality(merged)
    print(f"[data] Filas auditadas (crudo): {audit['n_rows']}, nulos: {audit['n_missing_total']}")
    print(f"[data] Filas del dataset final: {len(df)}")

    X = select_features(df.drop(columns=["Walc"]))
    y = (df["Walc"] >= 3).astype(int)  # target binarizado (consumo alto)
    split = stratified_split(X, y)
    print("[data] Split:", {k: len(v) for k, v in split.items() if k in ("X_train", "X_val", "X_test")})
    print(check_stratification(split["y_train"], split["y_val"], split["y_test"]))
    save_processed(**{k: v for k, v in split.items()})
    print("[data] OK -> data/processed")


def phase_eda():
    """Ejecuta el notebook de EDA y exporta a HTML (fase 4)."""
    from nbconvert import HTMLExporter
    from nbformat import read

    nb_path = PROJECT_ROOT / "notebooks" / "02_eda_exploracion.ipynb"
    out_path = PROJECT_ROOT / "reports" / "02_eda.html"
    if not nb_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el notebook de EDA en {nb_path}. "
            "Verifica que el notebook existe en notebooks/."
        )
    nb = read(nb_path, as_version=4)
    body, _ = HTMLExporter().from_notebook_node(nb)
    out_path.write_text(body, encoding="utf-8")
    print(f"[eda] Exportado -> {out_path}")


def phase_models():
    """Fases 11-16: baselines, modelado, CV, tuning y entrenamiento final."""
    import json

    import pandas as pd

    from src.data.load_data import load_processed
    from src.models.train_model import get_model_factories, run_cv
    from sklearn.model_selection import StratifiedKFold

    d = load_processed()
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for name, factory in get_model_factories().items():
        if name == "BaselineDummy":
            continue
        res = run_cv(X_train, y_train, name, factory, cv)
        results.append(res)
        print(f"[models] {name}: ROC-AUC CV = {res['mean']:.4f} +/- {res['std']:.4f}")

    # Modelo final: RandomForest tuneado (ganador de la fase 09)
    rf_params_path = PROJECT_ROOT / "configs" / "best_params_rf.json"
    if rf_params_path.exists():
        best = json.loads(rf_params_path.read_text(encoding="utf-8"))
        # Coherencia con la documentación: el modelo final usa class_weight balanceado
        best.setdefault("class_weight", "balanced")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from src.models.train_model import make_pipeline
        from src.evaluation.metrics import find_optimal_threshold
        import joblib

        pipeline = make_pipeline(RandomForestClassifier(**best))
        X_fit = pd.concat([X_train, X_val], axis=0)
        y_fit = pd.concat([y_train, y_val], axis=0)
        pipeline.fit(X_fit, y_fit)
        joblib.dump(pipeline, PROJECT_ROOT / "models" / "final_model.joblib")

        # Recalibración sigmoidal (Platt) sobre validation (coherente con notebook 10)
        from sklearn.linear_model import LogisticRegression
        p_val = pipeline.predict_proba(X_val)[:, 1]
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(p_val.reshape(-1, 1), y_val)
        joblib.dump(calibrator, PROJECT_ROOT / "models" / "final_calibrator.joblib")

        # Umbral de coste sobre probabilidades calibradas + metadatos
        proba_val_cal = calibrator.predict_proba(p_val.reshape(-1, 1))[:, 1]
        umbral = find_optimal_threshold(y_val, proba_val_cal)["optimal_threshold"]
        metadata = {
            "model_name": "RandomForest",
            "params": best,
            "features": list(X_fit.columns),
            "threshold": float(umbral),
            "random_state": 42,
            "n_train": int(len(X_fit)),
            "dataset_version": "uci-student-performance-2008",
            "target_definition": "Walc >= 3 (consumo alto fin de semana)",
            "preprocessing": "ColumnTransformer(OneHot + Ordinal + mediana), fit solo con train",
            "calibration": "Sigmoidal (Platt) ajustado sobre validation",
        }
        (PROJECT_ROOT / "models" / "final_model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[models] Modelo final RandomForest guardado -> models/final_model.joblib")
        print(f"[models] Calibrador guardado; umbral de coste = {umbral:.3f}")
    else:
        from src.models.train_model import train_final
        train_final(X_train, y_train, X_val, y_val, X_test, y_test,
                    model_name="LightGBM", use_val=True)
        print("[models] Entrenamiento final OK -> models/final_model.joblib")


def phase_api():
    """Fase 21: levanta la API para una prueba rápida de humo."""
    import uvicorn
    print("[api] Levantando API en http://127.0.0.1:8000 (Ctrl+C para salir)")
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)


def phase_tests():
    """Pruebas unitarias con pytest (fase 21.3)."""
    import pytest
    raise SystemExit(pytest.main(["-v", "tests/"]))


PHASES = {
    "data": phase_data,
    "eda": phase_eda,
    "models": phase_models,
    "api": phase_api,
    "tests": phase_tests,
}


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo Student Alcohol Consumption")
    parser.add_argument("--phase", choices=list(PHASES) + ["all"], default="all")
    args = parser.parse_args()
    if args.phase == "all":
        phase_data()
        phase_models()
        phase_tests()
    else:
        PHASES[args.phase]()


if __name__ == "__main__":
    main()
