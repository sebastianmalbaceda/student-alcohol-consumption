# Guía de contribución — Student Alcohol Consumption

¡Gracias por querer contribuir! Este proyecto sigue el índice maestro
`Indice-para-proyectos-de-ML-IA.md` y las buenas prácticas de ML reproducible.

## Flujo recomendado

1. **Abre un issue** describiendo el cambio (bug, mejora, feature).
2. **Haz un fork** y crea una rama: `git checkout -b fix/descripcion`.
3. **Implementa** siguiendo la estructura existente (`src/`, `configs/`, `notebooks/`).
4. **Añade o actualiza tests** en `tests/` para tu cambio.
5. **Ejecuta las comprobaciones**:

```bash
python -m pytest tests/ -v        # todos los tests
python -m compileall -q src scripts tests   # sintaxis
```

6. **Actualiza la documentación** (README, docs/, CHANGELOG) si el cambio afecta
   al comportamiento o a las métricas.
7. **Envía el PR** describiendo qué cambia y por qué.

## Reglas de oro

- **Nunca** modifiques `data/raw/` (inmutabilidad de datos).
- **Nunca** uses el test para decisiones de modelado (test bloqueado).
- Cualquier cambio en features, preprocessing o modelo debe pasar por CV y
  registrarse en `reports/experiments.csv`.
- Mantén la semilla maestra (42) y documenta cualquier cambio de protocolo.
- Si cambias el modelo final, actualiza `models/final_model_metadata.json`,
  `models/registry.json` y la model card.

## Estilo

- Python 3.11, `ruff` para linting (ver `pyproject.toml`).
- Celdas de notebook precedidas de markdown explicativo.
- Comentarios y docstrings en español (idioma del proyecto).

## Reportar vulnerabilidades

No abras issues públicos para problemas de seguridad: consulta `SECURITY.md`.
