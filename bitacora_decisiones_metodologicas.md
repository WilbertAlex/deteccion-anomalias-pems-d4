# Bitácora de decisiones metodológicas — PPI Detección de anomalías PeMS-D4

Este documento registra cada decisión de diseño tomada durante el preprocesamiento,
con su justificación y evidencia. Alimenta directamente la Sección 2.3 del protocolo
(Variables de entrada y preprocesamiento), la Sección 2.4 (Partición temporal) y la
Sección 2.11 (Reproducibilidad).

---

## Decisión 1 — Fuente e integridad del dataset
**Semana 1 (7–13 sep. 2026)**

- Fuente: `github.com/Davidham3/ASTGCN/blob/master/data/PEMS04/pems04.npz`
- Hash SHA-256 verificado (múltiples veces, entornos independientes):
  `95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a`
- Forma confirmada: `(16992, 307, 3)` — coincide con lo especificado en el Cuadro 4 del protocolo.
- **Estado: cerrada.**

---

## Decisión 2 — Valores faltantes: no existen como NaN en el archivo crudo
**Semana 1–2**

- Verificación directa: `np.isnan(raw).sum() == 0` y `np.isinf(raw).sum() == 0`.
- El problema real de calidad detectado no son NaN, sino valores congelados o en
  cero de forma sostenida (ver Decisión 3).
- **Estado: cerrada.**

---

## Decisión 3 — Exclusión de sensores de baja calidad
**Semana 1–2**

**Criterio adoptado:** se excluye una estación si tiene más del 20% de observaciones
de flujo en cero, o al menos una racha de 24 horas consecutivas con el mismo valor
exacto de flujo.

**Resultado:** 16 de 307 sensores excluidos (5.2%). 291 retenidos. Sin traslape
entre ambos criterios.

**Por qué exclusión y no interpolación:** interpolar una racha de 24+ horas
equivaldría a inventar casi un día completo de tráfico sintético.

- **Estado: cerrada.**

---

## Decisión 4 — Regla de hora pico
**Semana 2**

**Regla adoptada:** `is_peak_hour = 1` si la hora está en `{12, 13, 14, 15, 16, 17}`
(percentil 75 del perfil horario agregado), respaldada por el criterio de
percentiles del Highway Capacity Manual.

**Limitación reconocida:** meseta de mediodía-tarde en vez de patrón bimodal
clásico, probablemente por mezcla de sensores de ambos sentidos de circulación.

- **Estado: cerrada.**

---

## Decisión 5 — Partición temporal walk-forward
**Semana 3 (21–27 sep. 2026)**

**Regla aplicada** (Sección 2.4 del protocolo): último 20% cronológico como
conjunto de prueba final; el 80% restante dividido en 4 bloques iguales, que
forman 3 pliegues expansivos (el entrenamiento de cada pliegue siempre inicia
en el paso 0 y crece; la validación es el bloque inmediatamente siguiente).

**Resultado, verificado en máquina local:**
- Conjunto de prueba final: pasos `[13593, 16992)` → 17 feb – 28 feb 2018 (3399 pasos, 20.0%)
- Bloques pre-prueba: ~3398 pasos cada uno
- Pliegue 1: train `[0, 3398)` / val `[3398, 6796)`
- Pliegue 2: train `[0, 6796)` / val `[6796, 10194)`
- Pliegue 3: train `[0, 10194)` / val `[10194, 13593)`

**Verificación automática:** control por `assert` confirmó que ningún pliegue
traslapa su entrenamiento con su validación, y que ninguno toca el conjunto de
prueba final.

- **Estado: cerrada.** Ver `configs/particion_temporal.json`.

---

## Decisión 6 — Normalización z-score por pliegue
**Semana 3**

**Regla aplicada** (Sección 2.3): `z = (x - μ_train) / σ_train`, con μ y σ
estimados **únicamente** con el rango de entrenamiento de cada pliegue —
nunca con validación ni prueba. Se normalizan las 3 variables crudas (flujo,
ocupación, velocidad); las variables temporales cíclicas no se normalizan,
ya están acotadas por construcción.

**Resultado, verificado en máquina local** (media de flujo por pliegue):
| Pliegue | Media de flujo (train) |
|---|---|
| 1 | 209.97 |
| 2 | 210.40 |
| 3 | 211.83 |

Las medias difieren progresivamente entre pliegues (verificado por `assert`),
confirmando que cada uno usa su propio rango de entrenamiento y no el dataset
completo por error.

- **Estado: cerrada.** Ver `configs/normalizacion_zscore.json`.

---

## Decisión 7 — Construcción de ventanas temporales
**Semana 3**

**Decisión de ingeniería:** no se materializan ni se guardan en disco las
ventanas completas de los 3 pliegues (representarían varios GB para modelos
que aún no existen, cuyo entrenamiento inicia recién en la Semana 7). En su
lugar, se implementó y verificó una función reutilizable
(`src/ventanas.py::construir_ventanas`), que los scripts de entrenamiento de
G0–G7 importarán directamente.

**Verificación realizada sobre el Pliegue 1** (train + validación):
- Ventanas de entrenamiento: `(985617, 12, 8)`
- Ventanas de validación: `(988818, 12, 8)`
- Conteo exacto según fórmula esperada (verificado por `assert`)
- Sin valores NaN ni Inf
- Primera ventana coincide exactamente con corte manual de control
- Todas las ventanas de validación tienen su instante final dentro del rango correcto



---

## Decisión 8 — Esquema A: ventana móvil causal sobre flujo sin normalizar
**Semana 4 (28 sep. – 4 oct. 2026)**

**Contexto:** la Sección 2.5.1 del protocolo presentaba una contradicción entre la ecuación (1) — ventana móvil de 288 pasos, intrínsecamente causal — y una frase posterior que indicaba que la ventana se calculaba "exclusivamente con datos del entrenamiento del pliegue correspondiente". Bajo las dos reglas las etiquetas difieren en validación y prueba.

**Decisión adoptada:** se adopta la versión causal. Para cada instante `t`, `mu` y `sigma` se calculan sobre los 288 pasos inmediatamente anteriores (`t-288` a `t-1`) de la serie de **flujo sin normalizar**. Umbral `abs(z) > 3`. Los primeros 288 pasos de cada sensor (1 de enero, 00:00-23:55) quedan sin etiqueta y se excluyen de la evaluación.

**Por qué la versión causal:**
- No filtra información futura.
- Produce etiquetas idénticas en los 3 pliegues y en prueba.
- Simplifica la comparación entre esquemas.

**Configuración sellada:** `configs/config_esquema_A.json`.
**Estado:** cerrada, ejecutada por `semana4_esquema_A.py`.

---

## Decisión 9 — Esquema B: prevalencia estratificada y semilla fija de inyección
**Semana 4 (28 sep. – 4 oct. 2026)**

**Contexto:** la Sección 2.5.2 fijaba la prevalencia objetivo en 3% "dentro de cada bloque de evaluación", sin especificar si también se inyecta en entrenamiento ni cómo tratar la semilla.

**Decisión adoptada:**
- Prevalencia objetivo: **3% por bloque sensor x pliegue, en cada split** (entrenamiento, validación y prueba).
- Inyección **también en entrenamiento**, porque las arquitecturas supervisadas requieren ejemplos anómalos para ajustar sus fronteras de decisión. Excepción: el Autoencoder G4 se entrena **solo con ventanas normales del train**, conforme a su régimen no supervisado.
- **Semilla de inyección fija (42)**, idéntica para todas las arquitecturas y pliegues, para que las diferencias observadas se atribuyan al modelo y no al azar de la inyección.
- Distribución de tipos: 40% puntual, 30% contextual, 30% colectiva.
- Tipos: puntual (1 paso, magnitud 4-6 sigma), contextual (percentiles P5/P95 del perfil hora x día del train del pliegue), colectiva (3, 6 o 12 pasos, reducción a 20-40% del nivel previo).

**Configuración sellada:** `configs/config_esquema_B.json`.
**Estado:** cerrada, ejecutada por `semana4_esquema_B.py`.

---

## Próximas decisiones pendientes (Semana 5, 5 – 11 oct. 2026)
- [ ] Implementación y ajuste de G0 con 12 configuraciones
- [ ] Comparación de prevalencia, distribución y solapamiento A vs B (PE1)
- [ ] Estadísticas del bloque 80% para el reentrenamiento final

---

## Decisión 10 — Esquema B: un arreglo alterado por pliegue y semilla por bloque
**Semana 5 (5 – 11 oct. 2026)**

**Contexto:** `semana4_esquema_B.py` escribía los tres pliegues sobre un mismo arreglo. Como los rangos de train se solapan, lo guardado era la inyección del pliegue 3; los conteos reportados para los pliegues 1 y 2 no correspondían a lo guardado (pliegue 1 train: 30 161 reportado vs 30 186 en el archivo final). Además, con semilla 42 idéntica, train y val de igual longitud recibían las mismas posiciones relativas de anomalías (verificado).

**Decisión:** `semana5_esquema_B_por_pliegue.py` genera `esquema_B_pliegue{1,2,3}.npz` (train y val inyectados con el perfil estimado solo con el train de ese pliegue) y `esquema_B_final.npz` (bloque 80 % y prueba, perfil del bloque 80 %). Semilla = `[42, id_pliegue, id_split]`. `src/inyeccion_anomalias.py` no cambia.

**Controles (assert):** nada alterado fuera de los bloques; label 0 implica features intactas; prevalencia en [2,9 %, 3,2 %]; train distinto de val; segunda corrida idéntica.

**Resuelto:** el perfil contextual se calculaba agrupando los 291 sensores; el protocolo indica un perfil por sensor. Ver Decisión 12.

## Decisión 11 — G0: grilla y regla de selección fijadas antes de ver resultados
**Grilla:** ventanas {6, 11, 22, 44} x multiplicadores {2,0, 2,5, 3,0} = 12 configuraciones (rejilla completa).
**Regla:** el PR-AUC se calcula sobre el score continuo y no depende del multiplicador; por eso (1) w* = argmax de PR-AUC medio de validación (3 pliegues) y (2) k* = argmax de F1 medio de validación con w*. Una sola pasada sobre la prueba.
**Score:** |x_t - mu| / sigma con mu, sigma de los w pasos previos (t excluido); sigma < 1e-6 implica score 0.
**Resultado:** w* = 22 y k* = 3,0 en ambos esquemas. k* está en el borde de la grilla; no se amplía a posteriori (presupuesto máximo de 12 configuraciones).


## Decisión 12 — Esquema B: perfil contextual estimado por sensor
**Semana 5 (5 – 11 oct. 2026)**

**Contexto:** el PPI (2.5.2) define las anomalías contextuales con los percentiles P5/P95 del perfil hora × día **de cada sensor**. El código de la Decisión 10 los calculaba agrupando los 291 sensores en cada celda (hora × día), de modo que un sensor de bajo caudal recibía como referencia el P5/P95 de toda la autopista.

**Decisión:** el perfil se estima por sensor, solo con el entrenamiento del pliegue (o con el bloque 80 % en el bloque final). `src/inyeccion_anomalias.py` añade el parámetro `por_sensor` a `construir_perfil_hora_dia` (por defecto `False`, que reproduce exactamente la Decisión 10) y `configs/config_esquema_B.json` fija `"perfil_por_sensor": true`. Semillas, tipos, porcentajes y duraciones no cambian.

**Efecto verificado (misma máquina, mismos datos):**
- Las posiciones de las etiquetas son idénticas a las de la Decisión 10 (el generador consume la misma secuencia aleatoria); solo cambian los valores de flujo de las anomalías contextuales (6 698 en el pliegue 1; 16 663 en el bloque final).
- Mediana de |Δflujo| en esas anomalías: 252 → 119 veh/5 min (pliegue 1) y 255 → 132 (bloque final): las anomalías contextuales son menos extremas al ser relativas a cada sensor.
- G0 en el Esquema B: la selección no cambia (w* = 22, k* = 3,0). PR-AUC de validación (w = 22): 0,3627 → 0,3445. Prueba: PR-AUC 0,3617 → 0,3453; lift 11,8× → 11,3×; F1 0,332 → 0,323; AUC-ROC 0,797 → 0,792. El Esquema A no cambia.
- Comparación A vs B: sin cambios (depende de las posiciones, no de los valores).
- Los 50 controles de `semana5_validacion_pipeline.py` pasan; el modo `por_sensor=False` reproduce byte a byte el arreglo del pliegue 1 de la Decisión 10.
- Verificado de forma idéntica en el entorno del equipo (Esquema B, G0, 50/50 controles y comparación A vs B).

**Limitaciones a declarar:**
1. Con un perfil por sensor cada celda hora × día tiene pocas observaciones: 12 a 24 por sensor en el pliegue 1 (mediana 24), 60 a 72 en el pliegue 3 y 72 a 84 en el bloque final. Los P5/P95 del pliegue 1 son por tanto más ruidosos.
2. En 0,8 % (pliegue 1) y 1,2 % (bloque final) de las anomalías contextuales el valor inyectado es 0 porque el P5 del sensor en esa celda es 0; y en ~1 % la diferencia con el valor original es < 5 veh/5 min. No se filtraron a posteriori (no se modificó el diseño tras ver resultados); queda como limitación.
3. Los resultados de G0 sobre el Esquema B de la Decisión 10 (agrupado) quedan **superados** por los de esta decisión.

---

## Decisión 13 — G0: historial mayor que la ventana de las redes y análisis de sensibilidad prefijado
**Semana 6 (12 – 18 oct. 2026)**

**Contexto:** el PPI (§2.2, §2.5.2) presenta a G0 como comparable a las redes porque todas puntúan con los pasos previos de la ventana (12 pasos: t−11 a t). La grilla de la Decisión 11 incluye ventanas de 22 y 44 pasos, y la selección en validación eligió 22, casi el doble del historial que reciben las redes. Con 11 pasos (lo máximo que cabe en la ventana, porque t no entra en su propia referencia) G0 rinde menos.

**Decisión:** no se rehace la selección (la grilla y la regla se fijaron antes de ver resultados). La asimetría se declara en el §2.2 y en el Anexo A, y el resultado de G0 con 11 pasos se reporta como **análisis de sensibilidad fijado antes de entrenar las redes** (tabla `condicion = G0_w11_sensibilidad` en `results/eda/g0_por_bloque.csv`; `semana6_g0_ampliado.py`).

**Resultados (verificados en el entorno del equipo):**

| Esquema | G0 elegida (w = 22) | G0 sensibilidad (w = 11) |
|---|---|---|
| B, PR-AUC validación (873 bloques) | 0,3445 | 0,2926 |
| B, PR-AUC prueba (291 bloques) | 0,3453 | 0,2926 |
| A, PR-AUC validación (256 bloques) | 0,0427 | 0,0295 |
| A, PR-AUC prueba (84 bloques) | 0,0838 | 0,0599 |

**Consecuencia para H1:** el contraste de H1 se evalúa contra G0 con w = 22 (la más exigente) y se reporta también contra G0 con w = 11. Si una red supera a la primera, supera a ambas.

## Decisión 14 — Normalización bajo el Esquema B: estadísticas de la serie original
**Semana 6 (12 – 18 oct. 2026)**

**Contexto:** `configs/normalizacion_zscore.json` se calculó con la serie original. Bajo el Esquema B, el entrenamiento de cada pliegue contiene ~3 % de valores alterados, de modo que μ y σ de la serie inyectada difieren de los de la original.

**Medición** (`semana6_g0_ampliado.py`, serie inyectada del entrenamiento vs original):

| Bloque | Δμ flujo | Δσ flujo | Δμ velocidad | Δσ velocidad |
|---|---|---|---|---|
| Pliegue 1 | −0,36 % | +3,93 % | −0,86 % | +7,75 % |
| Pliegue 2 | −0,32 % | +4,05 % | −0,85 % | +7,55 % |
| Pliegue 3 | −0,31 % | +4,08 % | −0,85 % | +7,08 % |
| Final | −0,30 % | +4,08 % | −0,85 % | +6,60 % |

(La ocupación difiere hasta 0,92 % en μ y 1,35 % en σ.)

**Decisión:** se usan las estadísticas de la serie original (`normalizacion_zscore.json`) en ambos esquemas. Razones: (1) el protocolo las define sobre el entrenamiento del pliegue y la inyección es un procedimiento posterior sobre esa serie; (2) así ambos esquemas comparten la misma transformación y la comparación A vs B no se mezcla con un cambio de escala; (3) la regla no usa validación ni prueba, de modo que no hay fuga. **Limitación:** la diferencia no es despreciable en la desviación de la velocidad (hasta 7,75 %); se declara y no se corrige.

---

## Decisión 15 — H1 bajo el Esquema B multicanal y ponderación de clases (fijada antes de entrenar las redes)
**Semana 8 (26 oct. – 1 nov. 2026) — APROBADA el 3 de noviembre de 2026 (ratificacion del equipo; asistentes y aviso al asesor en el acta de esa reunion)**

**Contexto:** la inyección del Esquema B altera flujo, ocupación y velocidad con multiplicadores fijos (Decisiones 10 y 12), mientras G0 opera solo sobre flujo. Por instantes, el 74.7 % de las anomalías son colectivas, 14.6 % puntuales y 10.7 % contextuales.

**Observación exploratoria (no es resultado):** con la configuración inicial del Cuadro 6, entrenando solo 3 épocas sobre 1/40 de las ventanas del pliegue 1 del Esquema B y evaluando en la validación de ese pliegue (misma métrica que G0: PR-AUC medio por bloque, 270 bloques evaluables frente a 291 de G0):

| Modelo | PR-AUC medio por bloque |
|---|---|
| G0 (w = 22), referencia | 0.340 |
| Autoencoder (no supervisado) | 0.331 |
| LSTM, 8 canales | 0.869 |
| LSTM, solo flujo + variables temporales | 0.520 |
| CNN-1D, solo flujo + variables temporales | 0.132 |

(Entrenamiento mínimo, una sola semilla, solo el pliegue 1; sirve para formular una precaución, no para concluir.) Una red supervisada con acceso a ocupación y velocidad supera ampliamente a G0, mientras que el autoencoder, que no usa las etiquetas, queda al nivel de G0. Eso es compatible con que parte de la ventaja bajo el Esquema B provenga de aprender la **firma multicanal de la inyección** (las perturbaciones se aplican con multiplicadores fijos a ocupación y velocidad) y no de detectar anomalías reales.

**Decisión:**
1. H1 se contrasta como está definida (PR-AUC por bloque, Friedman/Holm), y **se reporta además el PR-AUC por tipo de anomalía** (puntual, contextual, colectiva) para todas las condiciones, con la composición por instantes.
2. **Ablación de canales (prefijada):** para cada arquitectura supervisada se repite la evaluación del Esquema B con solo flujo y variables temporales (ocupación y velocidad anuladas), con 1 semilla, y se reporta junto a la corrida principal. Es la comparación de entrada más cercana a G0, que solo ve flujo.
3. Toda conclusión sobre H1 bajo el Esquema B se declara **condicionada a la firma multicanal** de la inyección; la lectura bajo el Esquema A (sin inyección) se presenta con el mismo peso.
4. **Ponderación de clases:** la corrida principal no usa pesos en ningún esquema. Se agrega un **análisis de sensibilidad** con 1 semilla y pesos `n_neg / n_pos` calculados en el entrenamiento de cada pliegue, para el Esquema A (prevalencia 0.10 %) y el B (3 %). Se reporta sin reemplazar la corrida principal.
5. La regla se fija antes de entrenar y no se modifica en función de los resultados.

**Limitación:** el análisis por tipo no elimina la firma multicanal; solo la hace visible.

## Decisión 16 — Reducción declarada de la búsqueda de hiperparámetros (§2.7.1)
**Semana 8 (26 oct. – 1 nov. 2026) — APROBADA el 3 de noviembre de 2026 (ratificacion del equipo; asistentes y aviso al asesor en el acta de esa reunion)**

**Contexto (medición del 27 oct., `results/eda/cronometro_epoca_cpu-windows.json`):** pliegue 3 del Esquema B = 2 963 253 ventanas de entrenamiento = 46 301 pasos por época con lote 64. En CPU con TensorFlow 2.16.2 (sin GPU), con un LSTM de medición (no es la configuración del Cuadro 6): 5.5–6.4 ms por paso, 4.3–5.0 min por época. El diseño literal del §2.7.1 son 12 configuraciones × 3 pliegues × 3 semillas × 6 arquitecturas × 2 esquemas = **1 296 entrenamientos** más 36 reentrenamientos finales; con 30 épocas por entrenamiento, 77 a 89 días de cómputo continuo para la búsqueda.

**Decisión** (parámetros en `configs/config_busqueda.json`):
- 12 configuraciones por arquitectura y esquema, muestreadas con semilla fija (`semilla_muestreo`).
- **1 semilla (42)** durante la búsqueda; hasta **10 épocas** con parada temprana (paciencia 3) por PR-AUC de validación.
- **Submuestreo fijo de ventanas** en entrenamiento y parada temprana: solo instantes `t` múltiplos de 4 (determinista, sin azar).
- Métrica de selección: media sobre los 3 pliegues de la media de PR-AUC por bloque en validación, solo bloques con ambas clases. *(Confirmar contra el texto del §2.7.1.)*
- La configuración ganadora de cada arquitectura y esquema se **reentrena con todas las ventanas y 3 semillas (42, 43, 44)** sobre el bloque del 80 %, y es lo único que entra al contraste final.
- G0: grilla completa de 12 (Decisión 11). G7: configuración por defecto publicada.
- **Selección sobre validación completa:** la métrica de selección (PR-AUC medio por bloque) se calcula con **todas** las ventanas de validación de cada pliegue; el submuestreo de 1 de cada 4 se usa solo para entrenar y para la parada temprana durante la búsqueda.
- **Reentrenamiento final sin conjunto independiente:** el bloque del 80 % contiene la validación del pliegue 3, así que ahí no se usa parada temprana. El número de épocas del reentrenamiento final es la **mediana de las épocas hasta la mejor validación** de la configuración ganadora en sus 3 pliegues de la búsqueda, y se registra antes de reentrenar.
- **Un solo entorno para comparar:** todas las máquinas usan el mismo `requirements-lock.txt` (TensorFlow 2.16.2) y las mismas semillas, y cada corrida registra su hardware. Los **tiempos de entrenamiento** (VD6) solo se comparan entre arquitecturas si se midieron en la misma máquina; los de la búsqueda repartida entre máquinas no se usan para ese fin.

**Estimación con esta reducción** (misma medición): unas 51 horas por arquitectura y esquema para el LSTM de medición. Si el entrenamiento se pasa a GPU y se mide una aceleración, la reducción puede relajarse; cualquier relajación se decide **antes** de ver resultados de las redes y queda registrada en una decisión posterior.

**Cómputo adicional no incluido en las 51 horas** (extrapolación con el LSTM de medición, 5.7 min por época del bloque del 80 %, 30 épocas, unas 2.8 h por entrenamiento): los 36 reentrenamientos finales (~102 h), la ablación de canales de la Decisión 15 (5 arquitecturas supervisadas, ~14 h) y el análisis con pesos de clase (6 arquitecturas × 2 esquemas, ~34 h). Son órdenes de magnitud, no una medición de las seis arquitecturas.

**Limitaciones declaradas:** con 1 semilla, la selección de configuración lleva ruido de semilla; con submuestreo, la selección usa menos datos que el entrenamiento final; las ventanas consecutivas comparten 11 de 12 pasos, lo que reduce, pero no elimina, la pérdida de información.

## Fe de erratas de la Decisión 16 (registrada el 3 de noviembre de 2026; el texto original se conserva)
**Descubierta el 2 nov. 2026 al medir con `fit()`; registrada al ratificar el equipo.**
- La Decisión 16 dice «unas 51 horas por arquitectura y esquema». Esa cifra corresponde al **total de las 12 combinaciones** (6 arquitecturas × 2 esquemas): el plan literal de 77 a 89 días dividido por 36 (1 semilla en vez de 3, 10 épocas en vez de 30, 1 de cada 4 ventanas) da 51 a 59 h.
- Los extras que menciona (reentrenamientos finales ~102 h, ablación ~14 h, pesos de clase ~34 h) se calcularon con 30 épocas; el reentrenamiento final no pasa de 10.
- Presupuesto vigente (`semana9_cronometro_fit.py` y `semana9_presupuesto.py`, CPU Windows, pliegue 3, Esquema B, lote 64): unas **185 h, 7.7 días continuos**, sin validación por época, puntuación final ni G7. Ver `results/eda/presupuesto_computo.md`.

## Decisión 17 — Especificación de arquitecturas (Cuadro 6) y espacio de búsqueda
**Semana 8 (26 oct. – 1 nov. 2026) — APROBADA el 3 de noviembre de 2026 (ratificacion del equipo; asistentes y aviso al asesor en el acta de esa reunion)**

**Contexto:** el Cuadro 6 del PPI fija la configuración inicial de G1–G6 pero no todos los detalles de implementación, y el PPI exige registrar los rangos de la búsqueda en archivos de configuración sin enumerarlos. *(La especificación se tomó de la versión del PPI del 1 de septiembre; confirmar que no cambió.)*

**Decisión:**
1. **Configuración inicial** (`configs/config_modelos.json`, clave `inicial`), verificada con `assert` sobre la estructura de cada modelo (`semana7_humo_modelos.py`, 54 controles): G1 y G2, dos capas de 64 y 32 unidades, Adam 1e-3, dropout 0.2; G3, Conv1D de 32, 64 y 128 filtros, kernel 3, GlobalMaxPooling1D, AdamW; G4, Flatten (96) → 32 → 16 → 8 → simétrico → 96; G5, 2 bloques, 4 cabezas, embedding 64, feed-forward 128, AdamW 5e-4, dropout 0.1; G6, Conv1D 64 (kernel 3), LSTM 64, GRU 32, Adam 1e-3, dropout 0.2. Lote de 64 y parada temprana por PR-AUC de validación (en G4, por pérdida de validación sobre ventanas normales).
2. **Interpretaciones no fijadas por el Cuadro 6:** el dropout de las capas recurrentes es el argumento `dropout` de la capa; convoluciones con padding causal; Transformer con posición aprendida y GlobalAveragePooling1D; **weight decay de AdamW = 1e-4** (G3 y G5; el Cuadro 6 dice "decaimiento de pesos" sin valor); regularización L2 del autoencoder = 0 en la configuración inicial.
3. **Espacio de búsqueda** (clave `espacio`): vecindad de la configuración inicial en anchura de capas, dropout, tasa de aprendizaje, kernel y decaimiento de pesos. La **configuración 0 es siempre la del Cuadro 6**; las 11 restantes se muestrean con semilla fija sin repetirla. El presupuesto es de 12 configuraciones por arquitectura, igual que G0.
4. G4 usa como score el error cuadrático medio de reconstrucción y se entrena solo con ventanas de etiqueta 0; el percentil del umbral que menciona el PPI solo afecta a F1 y FPR, no a PR-AUC.

**Limitación:** los rangos son una propuesta del equipo; el PPI solo exige que queden registrados antes de la ejecución.

## Decisión 18 — Contraste primario de H1 y reglas estadísticas
**Semana 9 (2–8 nov. 2026) — APROBADA el 3 de noviembre de 2026 (ratificacion del equipo; asistentes y aviso al asesor en el acta de esa reunion)**

**Contexto:** el PPI (§1.6, §2.9, `config_final.json`) contrasta H1 con Friedman y Holm sobre los 873 bloques de validación; el short paper lo evaluaba sobre la prueba. Se fija antes de abrir la prueba final y antes de ver resultados de las redes (a la fecha no existe ninguna corrida de búsqueda).

**Decisión:**
1. **Conjunto de H1.** El contraste confirmatorio se hace sobre la **prueba final (291 bloques sensor), en una sola pasada**. Friedman, Holm contra G0 y Nemenyi sobre los 873 bloques de validación se reportan completos como análisis secundario. Razón: las configuraciones se eligen en la validación y su último bloque entra en el reentrenamiento final; la prueba es el único conjunto que ninguna decisión toca. Es una **desviación explícita del PPI** (§2.7.1, último párrafo) y se comunica al asesor antes de abrir la prueba.
2. **Margen.** El contraste primario no usa margen (δ = 0), como la H1 del PPI. Se reporta como criterio secundario de relevancia práctica si la diferencia media de PR-AUC frente a G0 supera 0.05.
3. **Prueba y semillas.** Wilcoxon de rangos con signo, alternativa unilateral de mejora, con Holm frente a G0 (α = 0.05). Familia: las comparaciones de G1–G6 (y G7 si entra) frente a G0 en el Esquema B, 6 comparaciones (7 con G7). Unidad pareada: el bloque sensor de la prueba (291). Las tres semillas se resumen por la **media** dentro de cada bloque; se conservan los resultados por semilla. Los bloques no son independientes (sensores vecinos, misma ventana temporal); se declara al reportar.
4. **Latencia (H2).** El criterio de ≤ 10 ms se decide con la **mediana**, sobre **una ventana (batch 1)**; se reportan p95 e IC 95 %. La medición principal usa el modelo **compilado** (`tf.function`); el modo eager se reporta como secundario. Se registran equipo, entorno, versiones, calentamiento (≥ 100) y repeticiones (≥ 1 000). Los fallos de compilación se registran sin excluir arquitecturas. G7 se mide en su propio framework y se declara como factor de confusión.
5. **Umbral operativo.** Se mantiene como en el §2.7.1 (paso 4): sobre el último bloque de validación del pliegue 3, sin usar la prueba. Como ese bloque forma parte del reentrenamiento del 80 %, el umbral puede ser optimista; afecta solo a F1 y FPR, no al PR-AUC primario. Si algún umbral llegara a usar información de la prueba, la declaración de limitación no basta y se revisa el carácter confirmatorio del análisis.
6. **H3 bajo el Esquema A.** Además del análisis principal, se reporta τ de Kendall y ρ de Spearman con PR-AUC **agregado por pliegue** (función `pr_auc_agregado`), como sensibilidad.
7. **Conteo de corridas.** El alcance mínimo es de **132 corridas** (108 de G1–G6, 6 evaluaciones de G0 y 18 de G7; 126 si G7 es determinista), como derivan la §3.1 y la nota de la Figura 2 del PPI; el 144 de la Figura 2 es una cifra anterior. Con la reducción de la Decisión 16 se ejecutan: 432 entrenamientos de búsqueda (6 × 2 × 3 pliegues × 12 configuraciones, 1 semilla), 36 reentrenamientos finales (6 × 2 × 3 semillas), las 6 evaluaciones de G0 y lo que quede de G7. En la validación, cada configuración tiene **una** semilla, no tres como dice el paso 1 del §2.7.1.
8. **G7.** Smoke-test en `venv_g7` con un límite de un día laborable; si no produce puntuaciones válidas sobre las ventanas en ese plazo, se activa la contingencia de 7 condiciones del §2.6.1 y se documenta el intento.

**Acciones derivadas:** corregir el PPI (§1.6, §2.9, Figura 2) y `config_final.json` en lo que cambian los puntos 1 y 7; poner el short paper de acuerdo; registrar la fecha del aviso al asesor.

## Decisión 19 — Entorno de entrenamiento y presupuesto
**Semana 9 (2–8 nov. 2026) — APROBADA el 3 de noviembre de 2026 (ratificacion del equipo; asistentes y aviso al asesor en el acta de esa reunion)**

1. **Entorno.** El entrenamiento y la latencia se hacen en **CPU, Windows**, con `requirements-lock.txt` (TensorFlow 2.16.2), lote 64 y las mismas semillas en todas las máquinas. La GPU bajo WSL2 no se usa para resultados (fue más lenta que la CPU con estos modelos). Los tiempos de entrenamiento (VD6) solo se comparan entre arquitecturas si se midieron en la misma máquina. `configs/hardware.json` actualizado el 2 nov. 2026.
2. **Presupuesto.** Ver la fe de erratas de la Decisión 16: unas 185 h (7.7 días continuos) con las seis arquitecturas medidas; costos por paso con `fit()`: autoencoder 2.55 ms, CNN-1D 4.43, LSTM 9.95, CNN-LSTM-GRU 10.68, GRU 12.85, Transformer 19.55. Para planificar se toma el doble (≈ 370 h) y se reparte la búsqueda entre las tres máquinas.
3. **Reducción de la búsqueda.** Se mantiene tal como está escrita (Decisión 16). Solo información de **tiempo** (costo medido de la validación por época en la primera corrida) puede motivar relajarla, nunca información de rendimiento; si se relaja, se aplica a **todas** las arquitecturas por igual y se decide antes de ver resultados de las redes.
4. **Registro del arnés.** `semana7_busqueda.py` guarda además `mejor_epoca` (época de la mejor validación) en cada entrenamiento, necesaria para la regla de épocas del reentrenamiento final de la Decisión 16. Es solo registro: no cambia el entrenamiento, los hiperparámetros ni las semillas.
5. **Confirmación contra el §2.7.1.** La métrica de selección (media sobre los 3 pliegues del PR-AUC por bloque de validación) es coherente con los pasos 1 y 2; se declaran dos matices: el PPI promedia sobre 3 semillas y la búsqueda usa 1, y el PPI no precisa si el PR-AUC de un pliegue es el promedio por bloque o el agregado (se usa el promedio por bloque).
