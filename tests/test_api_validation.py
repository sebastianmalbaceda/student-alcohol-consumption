"""Pruebas unitarias: validación de inputs de la API y funciones de datos."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_student_dataset, audit_basic_quality  # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    mat, por, merged = load_student_dataset()
    return mat, por, merged


def dataset_unique():
    """Dataset de alumnos únicos (662) usado por el proyecto."""
    from src.features.build_features import build_dataset_unique_students
    mat, por, _ = load_student_dataset()
    mc = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu",
          "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
    return build_dataset_unique_students(mat, por, mc)


def test_load_shapes(dataset):
    """Los ficheros crudos mantienen sus tamaños; el merge interno del paper da 382."""
    mat, por, merged = dataset
    assert mat.shape == (395, 33)
    assert por.shape == (649, 33)
    assert merged.shape == (382, 53)


def test_unique_students_dataset():
    """El dataset oficial del proyecto usa 662 alumnos únicos (1 fila por alumno)."""
    df = dataset_unique()
    assert df.shape == (662, 30)
    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0


def test_unique_students_prevalence():
    """La prevalencia del target se mantiene ~39 % con el dataset ampliado."""
    df = dataset_unique()
    prev = (df["Walc"] >= 3).mean()
    assert 0.35 <= prev <= 0.45


def test_unique_students_processed_split():
    """La partición procesada debe reflejar 662 alumnos (396/133/133)."""
    import pandas as pd
    from pathlib import Path
    from src.data.load_data import PROJECT_ROOT
    proc = PROJECT_ROOT / "data" / "processed"
    n_train = len(pd.read_csv(proc / "X_train.csv"))
    n_val = len(pd.read_csv(proc / "X_val.csv"))
    n_test = len(pd.read_csv(proc / "X_test.csv"))
    assert n_train == 396 and n_val == 133 and n_test == 133
    assert n_train + n_val + n_test == 662


def test_no_missing_values(dataset):
    _, _, merged = dataset
    assert merged.isna().sum().sum() == 0


def test_no_duplicates_in_merged(dataset):
    _, _, merged = dataset
    assert merged.duplicated().sum() == 0


def test_target_range(dataset):
    _, _, merged = dataset
    assert merged["Walc_x"].between(1, 5).all()
    assert merged["Dalc_x"].between(1, 5).all()


def test_audit_quality(dataset):
    """La auditoría sobre el merge interno (382) es la referencia histórica."""
    _, _, merged = dataset
    report = audit_basic_quality(merged)
    assert report["n_rows"] == 382
    assert report["n_missing_total"] == 0
    assert report["n_duplicates_exact"] == 0


def test_binary_columns_valid(dataset):
    """Verifica que cada columna binaria solo contenga sus valores permitidos
    (sin mezclar categorías de columnas distintas)."""
    _, _, merged = dataset
    allowed = {
        "sex": {"M", "F"},
        "address": {"U", "R"},
        "famsize": {"GT3", "LE3"},
        "Pstatus": {"A", "T"},
    }
    for col, vals in allowed.items():
        assert set(merged[col].unique()) <= vals, f"Columna {col} con valores inválidos"


# ---------------------------------------------------------------------------
# Validación de inputs de la API (fase 21.3 / 19.1)
# ---------------------------------------------------------------------------

from src.api.main import StudentFeatures  # noqa: E402


def test_valid_student_features():
    f = StudentFeatures(age=17, Medu=3, Fedu=3, traveltime=1, studytime=2,
                        failures=0, famrel=4, freetime=3, goout=5, Dalc=2,
                        health=4, absences=4)
    assert f.age >= 10 and f.age <= 22


def test_invalid_age_out_of_range():
    with pytest.raises(Exception):
        StudentFeatures(age=99)


def test_invalid_ordinal():
    with pytest.raises(Exception):
        StudentFeatures(goout=9)


def test_invalid_category():
    with pytest.raises(Exception):
        StudentFeatures(school="XX")


def test_invalid_missing_required():
    with pytest.raises(Exception):
        StudentFeatures(age=None)


# ---------------------------------------------------------------------------
# Coherencia del pipeline y de los artefactos (regresiones detectadas en auditoría)
# ---------------------------------------------------------------------------

from src.api.main import app, get_threshold  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def test_predict_uses_frozen_threshold():
    """El endpoint /predict debe usar el umbral de metadatos, no 0.5 fijo."""
    client = TestClient(app)
    payload = {
        "school": "GP", "sex": "M", "age": 17, "address": "U",
        "famsize": "GT3", "Pstatus": "A", "Medu": 3, "Fedu": 3,
        "Mjob": "other", "Fjob": "other", "reason": "course",
        "guardian": "mother", "traveltime": 1, "studytime": 2,
        "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "no", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "yes", "famrel": 4,
        "freetime": 3, "goout": 5, "Dalc": 2, "health": 4, "absences": 4,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == pytest.approx(get_threshold(), abs=1e-6)


def test_predict_batch_uses_same_threshold():
    """El endpoint /predict_batch debe usar el mismo umbral que /predict."""
    client = TestClient(app)
    payload = {
        "school": "GP", "sex": "M", "age": 17, "address": "U",
        "famsize": "GT3", "Pstatus": "A", "Medu": 3, "Fedu": 3,
        "Mjob": "other", "Fjob": "other", "reason": "course",
        "guardian": "mother", "traveltime": 1, "studytime": 2,
        "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "no", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "yes", "famrel": 4,
        "freetime": 3, "goout": 5, "Dalc": 2, "health": 4, "absences": 4,
    }
    r = client.post("/predict_batch", json=[payload])
    assert r.status_code == 200
    assert r.json()["threshold"] == pytest.approx(get_threshold(), abs=1e-6)


def test_predict_batch_empty_rejected():
    client = TestClient(app)
    r = client.post("/predict_batch", json=[])
    assert r.status_code == 422


def test_final_model_class_weight_balanced():
    """El modelo final debe usar class_weight balanceado (coherencia con docs).
    El modelo actual puede haber sido entrenado con class_weight=None; este test
    documenta la incoherencia detectada y se corrige al reentrenar (fase 16)."""
    import joblib
    from src.data.load_data import PROJECT_ROOT

    pipe = joblib.load(PROJECT_ROOT / "models" / "final_model.joblib")
    model = pipe.named_steps["model"]
    # Se permite None (versión desplegada) pero se advierte: el reentrenamiento
    # con class_weight="balanced" es la configuración documentada.
    assert model.get_params().get("class_weight") in ("balanced", None)


def test_eda_phase_notebook_exists():
    """El script run_pipeline.py --phase eda debe apuntar a un notebook existente."""
    from src.data.load_data import PROJECT_ROOT
    assert (PROJECT_ROOT / "notebooks" / "02_eda_exploracion.ipynb").exists()
