# ============================================================
# Makefile — Student Alcohol Consumption
# Comandos comunes para desarrollo, entrenamiento y despliegue
# ============================================================

PYTHON ?= python
PIP ?= pip

.PHONY: help install setup data eda models tests api batch lint clean

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Instala las dependencias del proyecto
	$(PIP) install -r requirements.txt

setup: ## Crea el entorno virtual e instala dependencias
	$(PYTHON) -m venv .venv
	$(PIP) install -r requirements.txt

data: ## Ejecuta la fase de datos (auditoría, split, preprocessing)
	$(PYTHON) scripts/run_pipeline.py --phase data

eda: ## Exporta el notebook de EDA a HTML
	$(PYTHON) scripts/run_pipeline.py --phase eda

models: ## Entrena y valida los modelos (CV + final)
	$(PYTHON) scripts/run_pipeline.py --phase models

tests: ## Ejecuta las pruebas unitarias
	$(PYTHON) -m pytest tests/ -v

api: ## Levanta la API REST local
	uvicorn src.api.main:app --reload --port 8000

batch: ## Ejemplo de inferencia batch
	$(PYTHON) scripts/batch_inference.py --input data/raw/archive/student-mat.csv --output reports/predicciones_ejemplo.csv

lint: ## Comprueba sintaxis de todos los Python
	$(PYTHON) -m compileall -q src scripts tests

clean: ## Limpia cachés y artefactos temporales
	rm -rf .pytest_cache __pycache__ src/**/__pycache__ tests/**/__pycache__
	rm -f reports/*.html reports/*.csv
