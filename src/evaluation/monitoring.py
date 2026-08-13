"""
Utilidades de monitorización y versionado de modelos (fase 23).

Centraliza:
- Cálculo de PSI (Population Stability Index) para detección de drift.
- Registro de versiones de modelo en models/registry.json (sin duplicados).
- Lectura de la zona de abstención definida en configs/config.yaml.
"""

from __future__ import annotations

import json
import pathlib
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index entre dos distribuciones.

    PSI = Σ (a_i - e_i) * ln(a_i / e_i), con e_i y a_i proporciones por bin.
    Regla habitual: PSI < 0.1 estable; 0.1-0.2 cambio moderado; > 0.2 cambio severo.
    """
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if expected.size == 0 or actual.size == 0:
        raise ValueError("psi() requiere arrays no vacíos")
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def load_config() -> dict:
    """Carga configs/config.yaml (con caché simple)."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_abstention_zone() -> List[float]:
    """Devuelve la zona de abstención configurada (p. ej. [0.30, 0.60])."""
    cfg = load_config()
    return cfg.get("monitoring", {}).get("abstention_zone", [0.30, 0.60])


def register_model_version(
    version: str,
    model_name: str,
    roc_auc_test: Optional[float] = None,
    notes: str = "",
    registry_path: pathlib.Path = REGISTRY_PATH,
) -> List[Dict]:
    """Añade una entrada al registro de versiones evitando duplicados.

    Si ya existe una entrada con la misma (version, model_name), la actualiza
    en lugar de duplicarla.
    """
    registry: List[Dict] = []
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    entry = {
        "version": version,
        "model": model_name,
        "date": date.today().isoformat(),
        "roc_auc_test": roc_auc_test,
        "status": "en_produccion",
        "notes": notes,
    }
    # Deduplicación por (version, model)
    registry = [
        e for e in registry
        if not (e.get("version") == version and e.get("model") == model_name)
    ]
    registry.append(entry)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def check_drift(
    expected: np.ndarray,
    actual: np.ndarray,
    feature_name: str,
    threshold: Optional[float] = None,
) -> Dict:
    """Evalúa el drift de una feature y devuelve un dict con el veredicto."""
    if threshold is None:
        cfg = load_config()
        threshold = cfg.get("monitoring", {}).get("drift_psi_threshold", 0.2)
    value = psi(expected, actual)
    return {
        "feature": feature_name,
        "psi": round(value, 4),
        "threshold": threshold,
        "alerta": bool(value > threshold),
    }
