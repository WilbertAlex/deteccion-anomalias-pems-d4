# Registro de cambios (Changelog)

Este proyecto sigue un versionado por semana de cronograma, no semver estricto,
dado que se trata de un proyecto de investigación con hitos definidos por el
protocolo (ver docs/cronograma_ppi_fechas.png).

## [0.3.0] — Semana 3 (21–27 sep. 2026) — cerrada
### Agregado
- Partición temporal walk-forward (3 pliegues expansivos + 20% prueba final)
- Normalización z-score por pliegue, ajustada solo con datos de entrenamiento
- Función reutilizable de construcción de ventanas (`src/ventanas.py`), verificada
  sobre el Pliegue 1 (985 617 ventanas de train, 988 818 de validación)

## [0.2.0] — Semana 2 (14–20 sep. 2026) — cerrada
### Agregado
- Criterio de exclusión de sensores (umbral 24h de racha congelada, o >20% ceros)
  documentado en `configs/config_preprocesamiento.json`
- Variables temporales derivadas: codificación cíclica de hora y día
- Regla de hora pico (percentil 75 + respaldo Highway Capacity Manual)
### Resultado
- 16 de 307 sensores excluidos (5.2%), 291 retenidos
- Verificado mediante control automático (`assert`) en `semana2_preprocesamiento.py`

## [0.1.0] — Semana 1 (7–13 sep. 2026) — cerrada
### Agregado
- Configuración inicial del entorno (Python, TensorFlow/Keras, `requirements.txt`)
- Descarga y verificación de integridad de PeMS-D4
  (hash `95a3c9b7...c999a`, verificado en 3 momentos independientes)
- Análisis exploratorio inicial (rangos, correlaciones de coherencia física)
