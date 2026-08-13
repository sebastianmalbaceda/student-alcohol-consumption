# ============================================================
# Dockerfile — API de inferencia Student Alcohol Consumption
# Uso: docker build -t student-alcohol-api . && docker run -p 8000:8000 student-alcohol-api
# ============================================================
FROM python:3.11-slim

# Evita la generación de bytecode y buffers de salida
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Primero las dependencias (aprovecha la caché de capas de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Después el código y los artefactos del modelo
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY data/raw/archive/student-mat.csv data/raw/archive/student-mat.csv
COPY data/raw/archive/student-por.csv data/raw/archive/student-por.csv

# Usuario no privilegiado (seguridad)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
