# Política de seguridad

## Versiones soportadas

| Versión | Soportada |
|---|---|
| 1.0.x | [OK] |

## Reportar una vulnerabilidad

Si encuentras un problema de seguridad (fuga de datos, inyección, acceso no
autorizado a la API, exposición de información personal, etc.):

1. **No** lo publiques en issues públicos.
2. Envía un correo al mantenedor con:
   - Descripción del problema y pasos para reproducirlo.
   - Impacto potencial.
   - Versión afectada.

## Respuesta

- Acuse de recibo: < 72 h.
- Plan de mitigación: < 7 días laborables.
- Divulgación coordinada tras la corrección.

## Consideraciones específicas del proyecto

- La API no maneja datos personales identificativos (dataset anónimo), pero valida
  rigurosamente los inputs (Pydantic) para evitar abusos.
- Los artefactos del modelo (`models/*.joblib`) no deben exponerse públicamente si
  contienen metadatos internos; se sirven solo a través de la API.
- El entorno y las claves viven en `.env` (ignorado por git); nunca se suben.
