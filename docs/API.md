# Documentación de la API — Student Alcohol Consumption

API REST de inferencia construida con **FastAPI** (fase 21 del guía metodológica).

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y carga del modelo |
| POST | `/predict` | Predicción individual (probabilidad + clase + etiqueta) |
| POST | `/predict_batch` | Predicción por lotes (máx. 100 registros) |
| GET | `/features` | Lista de columnas esperadas y rangos válidos |

Documentación interactiva (Swagger): `http://127.0.0.1:8000/docs`

## Arranque

```bash
# desde la raíz del proyecto (con .venv activado)
uvicorn src.api.main:app --reload --port 8000
```

## Ejemplo de petición

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "school": "GP", "sex": "M", "age": 17, "address": "U", "famsize": "GT3",
    "Pstatus": "A", "Medu": 3, "Fedu": 3, "Mjob": "other", "Fjob": "other",
    "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 2,
    "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "no", "nursery": "yes", "higher": "yes", "internet": "yes",
    "romantic": "yes", "famrel": 4, "freetime": 3, "goout": 5, "Dalc": 2,
    "health": 4, "absences": 4
  }'
```

## Respuesta

```json
{
  "probability": 0.87,
  "prediction": 1,
  "label": "alto",
  "threshold": 0.45,
  "features_used": 29
}
```

## Validación de inputs (fase 21.3)

- Tipos estrictos (int, str).
- Rangos: edad 10-22; ordinales 1-5 (o 0-4 en `failures`, 0-4 educación); `absences` 0-93.
- Categorías restringidas mediante `Literal` (p. ej. `school ∈ {GP, MS}`).
- Errores 422 con mensaje descriptivo; el pipeline nunca recibe datos inválidos.

## Notas

- El modelo debe existir en `models/final_model.joblib` (generado por el notebook 10
  o `scripts/run_pipeline.py --phase models`).
- El umbral de la respuesta se lee de `models/final_model_metadata.json`.
- Política de abstención: `0.30 <= probability < 0.60` -> recomendable revisión humana
  (definida en configs/config.yaml: monitoring.abstention_zone)
  (el campo `label` sigue devolviéndose, pero el cliente debe tratar esta zona como dudosa).
