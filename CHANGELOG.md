# Registro de cambios (Changelog)

Este proyecto sigue un versionado por semana de cronograma, no semver estricto,
dado que se trata de un proyecto de investigación con hitos definidos por el
protocolo (ver docs/cronograma_ppi_fechas.png).
## [0.9.0] — Semana 9 (2–8 nov. 2026) — en curso (corte: 2 nov.)
### Agregado
- `docs/short_paper/`: short paper en inglés y español (`paper.tex`, `paper_es.tex`, `references.bib`) con método y resultados preliminares de G0
- `semana9_cronometro_fit.py`: cronómetro con `fit()` de las seis arquitecturas (CPU Windows, pliegue 3, Esquema B, lote 64, `paso_t` = 4); resultados en `results/eda/cronometro_fit_*.json`
- `semana9_presupuesto.py`: presupuesto de cómputo calculado de esos JSON (`results/eda/presupuesto_computo.md`): 185 h, unos 7,7 días continuos, sin validación por época ni puntuación final
- `semana9_estadistica.py`: Friedman, Wilcoxon pareado con Holm frente a G0, Nemenyi exploratorio, τ de Kendall y ρ de Spearman con IC bootstrap (H3) y análisis complementario por pliegue; `--autoprueba` con datos simulados
- `semana9_latencia.py`: latencia con `perf_counter` (100 de calentamiento, 1 000 medidas, lotes 1, 291 y 307; mediana, p95 e IC 95 %); sin ejecutar aún sobre modelos entrenados
- `docs/borrador_decisiones_18_19.md`: borrador SIN APROBAR de la Decisión 18 (contraste primario de H1) y la 19 (entorno de entrenamiento y fe de erratas de la 16)
- Cronómetro de época de la semana 6 repetido en CPU Windows y en GPU bajo WSL2: la GPU resultó más lenta que la CPU con lote 64
### Corregido
- `configs/hardware.json`: registra que el entrenamiento y la latencia se hacen en CPU (fecha 2 nov.)
- Estimación de la Decisión 16: las «51 horas por arquitectura y esquema» eran el total de las 12 combinaciones; la corrección queda en el borrador de la Decisión 19
### Pendiente
- Aprobar y sellar las Decisiones 15–17 y `config_final.json`; decidir las 18 y 19; lanzar la búsqueda (`results/busqueda/` sigue vacía)
- Smoke-test de G7 en `venv_g7` o contingencia de 7 condiciones; medir latencia de los modelos entrenados; corregir el cuerpo del PPI

## [0.8.0] — Semana 8 (26 oct. – 1 nov. 2026) — cerrada
Todo el trabajo consta en commits del 27 de octubre.
### Agregado
- `src/datos.py`: cargador `tf.data` de ventanas (batch, 12, 8) por esquema y pliegue; `semana6_verificar_datos.py` (73 controles)
- `src/entrenar.py`: semillas, parada temprana por PR-AUC y evaluación por bloque
- `semana6_cronometro_epoca.py`: costo real de una época (CPU Windows, LSTM de medición, lote 64: 4,3 a 5,0 min por época del pliegue 3); la búsqueda literal del PPI se estimó en 77 a 89 días continuos
- `src/modelos.py`: G1–G6 según el Cuadro 6; `semana7_humo_modelos.py`: prueba de humo de los modelos
- `semana7_busqueda.py`, `configs/config_modelos.json`, `configs/config_busqueda.json`: búsqueda acotada de 12 configuraciones por arquitectura
- Decisiones 15 (H1 multicanal y ablación de canales), 16 (reducción de la búsqueda y reentrenamiento final) y 17 (especificación del Cuadro 6 y espacio de búsqueda) en la bitácora, SIN APROBAR
- Informe de la Semana 7
### Desviaciones
- El gate de la Semana 7 (sellado de `config_final.json`) sigue sin cumplirse
- Varios commits del 27 de octubre llevan el rótulo `semana6` o `semana7` aunque el trabajo ocurrió en la Semana 8

## [0.7.0] — Semana 7 (19–25 oct. 2026) — sin actividad registrada
- Ningún commit entre el 13 y el 27 de octubre. El gate de la Semana 7 no se cumplió y se trasladó a la Semana 8 (ver `docs/informes/semana7_por_integrante.pdf`)

## [0.6.0] — Semana 6 (12–18 oct. 2026) — cerrada (el cargador y el cronómetro previstos aquí se hicieron en la Semana 8)
### Agregado
- `semana6_g0_ampliado.py`: G0 por bloque (`g0_por_bloque.csv`), umbral de recall 0,90, PR-AUC por tipo de anomalía y normalización bajo el Esquema B
- `esquema_B_*.npz` incluye la clave `tipos` (puntual, contextual, colectiva)
- Decisiones 13 (G0 con historial mayor que la ventana de las redes; sensibilidad con 11 pasos) y 14 (normalización bajo B con la serie original)

## [0.5.0] — Semana 5 (5–11 oct. 2026) — cerrada
### Agregado
- `semana5_esquema_B_por_pliegue.py`: Esquema B por pliegue (Decisión 10)
- `src/g0_linea_base.py`, `semana5_g0.py`, `configs/config_g0.json`: G0 con 12 configuraciones (Decisión 11)
- `semana5_validacion_pipeline.py`: 50 controles de fuga temporal por perturbación
- `semana5_comparacion_A_vs_B.py`: comparación PE1 sobre bloques de evaluación
- `auditar_repo.py`, `results/eda/LEEME_resultados.md`, informes de las Semanas 4 y 5
### Corregido
- Esquema B v1: arreglo único para tres pliegues y misma semilla en train y val (superado)
- Texto del Anexo: solo el 29,2 % de los bloques tiene ambas clases bajo el Esquema A
### Pendiente
- Control de integración de G7, correcciones del protocolo, decisión del perfil contextual por sensor

## [0.4.0] — Semana 4 (28 sep. – 4 oct. 2026) — cerrada
### Agregado
- Esquema A: etiquetado proxy causal (ventana móvil de 288 pasos, umbral |z|>3, flujo sin normalizar)
  - Prevalencia global observada: 0.10% (hallazgo empírico por debajo del rango esperado 0.3%-2%)
  - Config: `configs/config_esquema_A.json`, reporte: `results/eda/prevalencia_esquema_A.json`
- Esquema B: inyección controlada de anomalías (3% por bloque sensor × pliegue, semilla fija 42)
  - Prevalencia global: 3.035%, consistente en los 7 splits evaluados
  - Config: `configs/config_esquema_B.json`, reporte: `results/eda/prevalencia_esquema_B.json`
  - Módulo reutilizable: `src/inyeccion_anomalias.py`
- Comparación A vs B para PE1 (prevalencia, Jaccard, bloques con ambas clases)
  - Reportes: `results/eda/comparacion_A_vs_B.json` y `.md`
- Normalización del bloque 80% inicial (para reentrenamiento final §2.7.1)
  - Añadida entrada `"final"` en `configs/normalizacion_zscore.json`
- Config `config_final.json` con decisiones pre-gate (bloques Friedman 873 val/291 test, batch 291/307)
- Decisiones 8 y 9 registradas en la bitácora metodológica
- `configs/hardware.json` con especificaciones reales del entorno de benchmark

### Corregido
- `config_preprocesamiento.json` movido de la raíz a `configs/`
- `.gitignore` activo (antes era `gitignore` sin punto)
- Datos crudos, procesados y resultados de EDA sacados del índice git
- README actualizado con estructura real y 5 scripts del pipeline
- CI mejorado: compila `src/` además de la raíz, valida todos los `configs/*.json`, verifica que datos y `venv` no estén versionados
- `requirements.txt` reestructurado (PyTorch en venv aislado para G7)

---

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
