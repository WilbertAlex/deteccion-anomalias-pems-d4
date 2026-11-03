# Evaluación comparativa de arquitecturas de Deep Learning para detección de anomalías en flujo vehicular (PeMS-D4)

Proyecto de Investigación IV — Escuela Profesional de Ingeniería de Sistemas, Universidad Peruana Unión (UPeU), filial Juliaca.

**Estado actual (2 nov. 2026):** Semana 9 de 12 en curso. Datos, Esquemas A y B, G0 y 50 controles de fuga verificados; cargador de datos, modelos G1–G6 (Cuadro 6), búsqueda acotada y cronómetros con `fit()` programados y verificados, con presupuesto de cómputo calculado (unas 185 h en CPU Windows). **Todavía no se ejecutó ninguna búsqueda de hiperparámetros**; las Decisiones 15–17 siguen sin aprobar y G7 sin smoke-test. Detalle en `CHANGELOG.md`.

## Qué es este proyecto

Comparamos 6 arquitecturas de Deep Learning (LSTM, GRU, CNN-1D, Autoencoder, Transformer, híbrido CNN-LSTM-GRU), una línea base estadística no paramétrica y un modelo publicado de referencia (SI-DMSeq), para detectar anomalías en series de flujo vehicular del dataset **PeMS-D4** (307 sensores originales → **291 efectivos** tras control de calidad, Área de la Bahía de San Francisco, enero–febrero 2018).

El protocolo evalúa las 8 condiciones bajo **dos esquemas independientes de etiquetado de anomalías** (proxy estadístico causal e inyección controlada), para analizar si el ranking de arquitecturas depende de cómo se define una anomalía — un vacío metodológico identificado en la literatura revisada.

## Integrantes

| Integrante | Código | Rol principal en el pipeline |
|---|---|---|
| Mayta Arotaype, Wilbert Alex | 202212463 | Preprocesamiento, calidad de datos, análisis de sensibilidad al etiquetado |
| Prieto Quispe, Juan Enders | 202210704 | Etiquetado, línea base G0, benchmarking computacional |
| Choquechambi Luque, Wrangler | 202220535 | Partición temporal, inyección de anomalías, análisis estadístico |

**Asesor:** Dr. Esteban Tocto Cano

## Estructura del repositorio

```
.
├── configs/                              (criterios versionados)
│   ├── config_preprocesamiento.json      (exclusion de sensores, hora pico)
│   ├── particion_temporal.json           (3 pliegues walk-forward + prueba final)
│   ├── normalizacion_zscore.json         (mu y sigma por pliegue + bloque final)
│   ├── config_esquema_A.json             (Esquema A causal, Decision 8)
│   ├── config_esquema_B.json             (Esquema B, Decisiones 9, 10 y 12)
│   ├── config_g0.json                    (G0: grilla y regla de seleccion, Decision 11)
│   ├── config_modelos.json               (G1-G6: configuracion inicial del Cuadro 6 y espacio de busqueda; PROPUESTA)
│   ├── config_busqueda.json              (reglas de busqueda y reentrenamiento final; PROPUESTA)
│   ├── config_final.json                 (borrador; el gate de la semana 7 no se cumplio)
│   └── hardware.json                     (caracterizacion del entorno; entrenamiento en CPU)
├── data/                                 (NO versionado; se regenera)
├── results/eda/                          (reportes .json/.md versionados; ver LEEME_resultados.md)
├── docs/
│   ├── imagenes/                         (figuras del PPI)
│   ├── informes/                         (cierres semanales por integrante, PDF)
│   ├── paper/                            (figuras del articulo corto)
│   ├── short_paper/                      (short paper en ingles y español: .tex, .pdf, references.bib)
│   ├── borrador_decisiones_18_19.md      (borrador SIN APROBAR de las Decisiones 18 y 19)
│   └── justificacion_dataset_pems04.pdf
├── scripts_paper/                        (generacion de figuras del articulo)
├── src/
│   ├── ventanas.py                       (ventaneo reutilizable)
│   ├── inyeccion_anomalias.py            (modulo del Esquema B)
│   ├── g0_linea_base.py                  (G0: score causal y evaluacion por bloque)
│   ├── datos.py                          (cargador tf.data de ventanas por esquema y pliegue)
│   ├── entrenar.py                       (semillas, parada temprana por PR-AUC, evaluacion por bloque)
│   └── modelos.py                        (G1-G6 segun el Cuadro 6)
├── eda_pems04.py                         (Semana 1)
├── semana2_preprocesamiento.py           (Semana 2)
├── semana3_particion.py / semana3_normalizacion.py / semana3_ventanas.py   (Semana 3)
├── semana4_esquema_A.py                  (Semana 4, vigente)
├── semana4_normalizacion_final.py        (Semana 4, vigente)
├── semana4_esquema_B.py                  (Semana 4, SUPERADO por semana5_esquema_B_por_pliegue.py)
├── semana4_comparacion_A_vs_B.py         (Semana 4, SUPERADO por semana5_comparacion_A_vs_B.py)
├── semana5_esquema_B_por_pliegue.py      (Semana 5)
├── semana5_g0.py                         (Semana 5)
├── semana5_validacion_pipeline.py        (Semana 5)
├── semana5_comparacion_A_vs_B.py         (Semana 5)
├── semana6_g0_ampliado.py                (Semana 6: G0 por bloque, recall 0.90, tipos, normalizacion)
├── semana6_verificar_datos.py            (Semana 6: 73 controles del cargador de datos)
├── semana6_cronometro_epoca.py           (Semana 6: costo real de una epoca y de la busqueda)
├── semana7_humo_modelos.py               (Semana 7-8: prueba de humo de los modelos G1-G6)
├── semana7_busqueda.py                   (Semana 7-8: busqueda acotada de hiperparametros; sin ejecutar aun)
├── semana9_cronometro_fit.py             (Semana 9: costo por paso con fit() de las 6 arquitecturas)
├── semana9_presupuesto.py                (Semana 9: presupuesto de computo calculado de los cronometros)
├── semana9_estadistica.py                (Semana 9: Friedman, Holm, Nemenyi, tau de Kendall; --autoprueba)
├── semana9_latencia.py                   (Semana 9: latencia con perf_counter, batch 1/291/307)
├── auditar_repo.py                       (auditoria de lo versionado)
├── bitacora_decisiones_metodologicas.md  (Decisiones 1-17; las 15-17 sin aprobar)
├── CHANGELOG.md, CITATION.cff, LICENSE, requirements.txt, requirements-lock.txt
└── README.md
```

**Orden de ejecucion:** `eda_pems04.py` → `semana2_preprocesamiento.py` → `semana3_particion.py` → `semana3_normalizacion.py` → `semana4_normalizacion_final.py` → `semana4_esquema_A.py` → `semana5_esquema_B_por_pliegue.py` → `semana5_g0.py` → `semana5_validacion_pipeline.py` → `semana5_comparacion_A_vs_B.py` → `semana6_g0_ampliado.py`. Después, las verificaciones y cronómetros de las semanas 8 y 9 (ver la sección 3).

## Reproducibilidad

### 1. Entorno

```bash
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Dataset

El dataset **no se versiona en este repositorio** (32 MB; su integridad se verifica por hash, no por control de versiones). Descárgalo y verifícalo:

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/pems04.npz" -OutFile "data/raw/pems04.npz"
certutil -hashfile data/raw/pems04.npz SHA256
```

**Linux / macOS:**
```bash
curl -sL -o data/raw/pems04.npz https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/pems04.npz
sha256sum data/raw/pems04.npz
```

El hash debe coincidir con:

```
95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a
```

Fuente original: Song, C., Lin, Y., Guo, S., & Wan, H. (2020). *Spatial-Temporal Synchronous Graph Convolutional Networks*. AAAI 2020. Ver justificación completa en [`docs/justificacion_dataset_pems04.pdf`](docs/justificacion_dataset_pems04.pdf).

### 3. Ejecutar el pipeline, en orden

```bash
python eda_pems04.py
python semana2_preprocesamiento.py
python semana3_particion.py
python semana3_normalizacion.py
python semana3_ventanas.py
python semana4_normalizacion_final.py
python semana4_esquema_A.py
python semana5_esquema_B_por_pliegue.py
python semana5_g0.py
python semana5_validacion_pipeline.py
python semana5_comparacion_A_vs_B.py
python semana6_g0_ampliado.py
```

Verificaciones y mediciones de las semanas 8 y 9 (se ejecutan después de generar los datos procesados):

```bash
python semana6_verificar_datos.py                  # cargador de datos: 73 controles
python semana7_humo_modelos.py                     # modelos G1-G6 contra el Cuadro 6
python semana9_cronometro_fit.py --etiqueta <etiqueta> --arq lstm --spe 1   # una corrida por arquitectura
python semana9_presupuesto.py                      # presupuesto de computo
python semana9_estadistica.py --autoprueba         # prueba del analisis estadistico con datos simulados
```

Cada script documenta sus decisiones en `configs/` y verifica sus propios resultados mediante controles automáticos (`assert`) antes de continuar.

### 4. Notas sobre PyTorch (G7)

PyTorch se instala **por separado en un entorno aislado** (`venv_g7`) para el smoke-test de G7 (SI-DMSeq), según la Sección 2.6.1 del PPI. El entorno principal del proyecto usa TensorFlow/Keras para G0–G6.

### 5. Entorno de entrenamiento

El entrenamiento y la latencia se hacen **en CPU, Windows**, con `requirements-lock.txt` (TensorFlow 2.16.2), lote 64 y las mismas semillas en todas las máquinas. La GPU bajo WSL2 resultó más lenta que la CPU con estos modelos y no se usa para resultados. Hardware documentado en `configs/hardware.json`.

## Documentación de avance

Los informes de cierre semanales, organizados por integrante, están en `docs/informes/`:

- Semana 1 — Preparación de datos
- Semana 2 — Calidad de datos y variables temporales
- Semana 3 — Partición, normalización y ventanas (cerrada y verificada)
- Semana 4 — Esquemas de etiquetado A y B (cerrada y verificada)
- Semana 5 — informe por integrante
- Semana 6 — informe por integrante
- Semana 7 — gate de configuración (sin actividad registrada)
- Semana 8 — cargador de datos, modelos G1–G6, búsqueda acotada y Decisiones 15–17

## Licencia y datos

**El código** se publica bajo licencia MIT (ver `LICENSE`). Esta licencia **no se extiende al dataset PeMS-D4**, que proviene del sistema Performance Measurement System (PeMS) del Departamento de Transporte de California (Caltrans) y se rige por sus propios términos de acceso y uso. Ver la nota de licencia en `docs/justificacion_dataset_pems04.pdf`.

## Cita

Si este trabajo o su metodología les resulta útil, por favor citen:

> Mayta Arotaype, W. A., Prieto Quispe, J. E., & Choquechambi Luque, W. (2026). *Evaluación comparativa de arquitecturas de Deep Learning frente a una línea base no paramétrica para la detección de anomalías en series de flujo vehicular en autopistas (PeMS-D4)*. Universidad Peruana Unión.