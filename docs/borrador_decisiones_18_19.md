# BORRADOR para la reunión del 3 de noviembre — Decisiones 18 y 19

> Estado: **PROPUESTA, sin aprobar.** Las casillas `[ ]` las marca el equipo en la reunión. Este borrador no decide por ustedes; solo ordena las opciones, qué cambia con cada una y qué documento hay que corregir. Fecha de redacción: 2 nov 2026. Hallazgos de origen: H-02 y H-04 de la revisión del 2 nov.

---

## Decisión 18 — Contraste primario de H1 y reglas estadísticas que dependen de ella

**Por qué hay que decidirla.** El PPI (§1.6, §2.9, `config_final.json`) contrasta H1 con Friedman y Holm sobre los **873 bloques de validación** y deja la prueba como confirmación. El short paper la evalúa sobre los **291 bloques de prueba** con un margen δ = 0.05. Los dos documentos no pueden decir cosas distintas. Debe quedar fijada **antes de abrir la prueba final y antes de ver resultados de las redes**.

### 18.1 Conjunto donde se contrasta H1

| Opción | Qué implica | Lo que hay que corregir después |
|---|---|---|
| `[ ]` **A. Validación (873 bloques), prueba como confirmación** — como está en el PPI | Más bloques (más potencia). Pero las configuraciones de las redes (12 por arquitectura) y la de G0 (12) se eligieron sobre esos mismos datos, y el reentrenamiento final de las redes incluye el último bloque de validación. La comparación queda expuesta a sesgo de selección; es simétrica en número de configuraciones, no en grados de libertad. | Short paper (Métodos, Hipótesis y lista de pendientes). |
| `[ ]` **B. Prueba (291 bloques), una sola pasada** — como está en el short paper | Estimación sin sesgo de selección. Menos bloques. La prueba se abre una vez, con las reglas ya selladas. Friedman/Nemenyi sobre validación pasan a ser exploratorios. | PPI §1.6, §2.9 y `config_final.json`. |

### 18.2 Resto de elecciones (cada una se marca por separado)

| Tema | Opciones | Nota |
|---|---|---|
| Margen mínimo δ de H1 | `[ ]` δ = 0.05 de PR-AUC (propuesto en el short paper) · `[ ]` δ = 0 (solo significancia) · `[ ]` otro valor: ____ | Con G0 en 0.345 de PR-AUC (Esquema B, prueba, w = 22), δ = 0.05 es un 14 % relativo. El script `semana9_estadistica.py` lo recibe como `--delta`. |
| Prueba pareada | `[ ]` Wilcoxon de rangos con signo (implementada) · `[ ]` t pareada · `[ ]` otra: ____ | PR-AUC por bloque no tiene por qué ser normal; el PPI es no paramétrico. |
| Resumen de las 3 semillas por bloque | `[ ]` media · `[ ]` mediana | Antes de contrastar, para tener un valor por bloque y condición. |
| Corrección de múltiples comparaciones | Holm contra G0 (ya en el PPI) | Sin cambio. |
| Criterio de H2 (≤ 10 ms) | `[ ]` mediana · `[ ]` percentil 95 | El PPI reporta mediana, p95 e IC 95 %, pero no dice cuál decide. |
| **Modo de llamada del modelo en la medición de latencia** | `[ ]` eager `modelo(x, training=False)` · `[ ]` compilado con `tf.function` · `[ ]` reportar ambos y decidir con el eager | **Dato nuevo (prueba en el sandbox de Claude, no en su laptop, solo para ver el orden de magnitud):** la diferencia entre ambos modos fue de 10 a 40 veces. En esa prueba el LSTM con batch 1 dio ~51 ms en eager y ~1.4 ms con `tf.function`; con el criterio ≤ 10 ms, **la conclusión de H2 cambiaría según el modo**. Hay que fijarlo antes de medir en su máquina. |
| Fuga del umbral | `[ ]` documentar como limitación · `[ ]` reservar un bloque distinto para el umbral | El reentrenamiento con el 80 % incluye el último bloque de validación donde se elige el umbral. |

### 18.3 Análisis complementario de H3 bajo el Esquema A (hallazgo H-06)

Con 71, 102 y 83 bloques evaluables por pliegue, y pocos positivos por bloque, es probable que el IC de τ cruce 0.7. Propuesta, para fijar ahora y no después de ver resultados: reportar además τ y ρ con **PR-AUC agregado por pliegue** (todos los sensores juntos, ya calculado para G0 en `g0_agregado.json`), como sensibilidad y sin sustituir el análisis principal. `[ ]` aprobado · `[ ]` no.

### 18.4 Registro
Elección del equipo, fecha y asistentes: ______________________. Después de marcarla: corregir el documento que quede en desacuerdo (ver tabla 18.1) y sellar `config_final.json` con fecha.

---

## Decisión 19 — Entorno de entrenamiento y fe de erratas de la estimación de la Decisión 16

### 19.1 Entorno
- Se entrena en **CPU, Windows**, con `requirements-lock.txt` (TensorFlow 2.16.2), lote 64 y las mismas semillas en todas las máquinas.
- La GPU bajo WSL2 **no se usa** para resultados: en la medición del LSTM de medición fue más lenta que la CPU (12 frente a 4.3–5.3 min por época). No se usan tiempos de máquinas distintas para comparar arquitecturas (VD6).
- Hay que actualizar `configs/hardware.json` (hoy con fecha del 6 de octubre y sin registrar CPU ni RAM de entrenamiento).

### 19.2 Fe de erratas de la Decisión 16 (el texto original se conserva; esto lo corrige)
**Lo que dice la Decisión 16:** «unas 51 horas por arquitectura y esquema».

**Lo que se comprobó:** las 51 horas corresponden al **total de las 12 combinaciones** (6 arquitecturas × 2 esquemas), no a cada una. Se comprueba con el propio cronómetro: la búsqueda literal del PPI se estimó en 77–89 días, y la reducción de la Decisión 16 la divide por 36 (1 semilla en vez de 3, 10 épocas en vez de 30, 1 de cada 4 ventanas): 77–89 días ÷ 36 = 51–59 h, es decir, el total. Por combinación son unas 7.7 h para el LSTM del Cuadro 6.

**Segundo error:** los extras de la Decisión 16 (reentrenamientos finales ~102 h, ablación ~14 h, pesos de clase ~34 h) se calcularon con **30 épocas**, pero el reentrenamiento final no pasa de 10.

**Medición nueva** (`semana9_cronometro_fit.py`, CPU Windows, LSTM del Cuadro 6, pliegue 3, Esquema B, lote 64, `paso_t` = 4): `train_on_batch` 9.12 ms/paso; `fit()` 9.95 ms/paso; `steps_per_execution` no acelera en CPU (8: 9.91; 32: 12.70). Producir lotes cuesta 0.39 ms/paso.

**Cifras actualizadas:** las genera `semana9_presupuesto.py` desde los JSON de cronómetro (ver `results/eda/presupuesto_computo.md`). Pegar aquí la tabla cuando se tengan las seis arquitecturas medidas. Con solo el LSTM, la cota es ≈ 180 h; la revisión del 2 nov, con cuatro arquitecturas medidas, extrapola ≈ 185 h. Ambas excluyen validación por época, puntuación final, G7 y las arquitecturas sin cronometrar.

**Corrección a un cálculo propio:** el factor del bloque del 80 % respecto al pliegue 3 es 13 593 / 10 194 = **1.333** (de `particion_temporal.json`), no 1.23. Sube unas 7 h el total calculado con el costo del LSTM (de ≈176 a ≈183 h).

**Consecuencia:** el presupuesto cabe en una sola máquina (648 h disponibles hasta el 29 de noviembre). La reducción de la búsqueda **puede mantenerse**; relajarla (más semillas, más ventanas) es una decisión del equipo que debe tomarse **antes de ver resultados de las redes**: `[ ]` mantener la reducción · `[ ]` relajar a: ______________.

### 19.3 Registro
Elección, fecha y asistentes: ______________________.
