# Decisiones 18 y 19 — ELECCIONES PROPUESTAS para aprobación (reunión del 3 de noviembre de 2026)

> **Estado: PROPUESTA, sin aprobar.** Las elecciones marcadas `[x]` las propuso el asistente de análisis a pedido del autor, con las razones que se indican. **No son decisiones del equipo hasta que el equipo las apruebe y quede constancia con fecha y asistentes** (ver 18.4 y 19.3). Las que cambian el texto del PPI (18.1) requieren **informar al asesor antes de abrir la prueba final**. Hallazgos de origen: H-02 y H-04 de la revisión del 2 de noviembre.

---

## Decisión 18 — Contraste primario de H1 y reglas estadísticas

### 18.1 Conjunto donde se contrasta H1
- `[ ]` A. Validación (873 bloques), prueba como confirmación — como está en el PPI.
- `[x]` **B. Prueba (291 bloques), una sola pasada, con las reglas ya selladas.** Friedman, Holm y Nemenyi sobre los 873 bloques de validación se reportan completos como análisis secundario.

**Razones.** (1) Las configuraciones de G0 y de las redes se eligieron en la validación, y el último bloque de validación entra en el reentrenamiento final; contrastar H1 ahí mezcla selección y evaluación. (2) La prueba es el único conjunto que ninguna decisión del estudio toca antes del contraste. (3) El costo es menos bloques (291 frente a 873) y que el PPI (§1.6, §2.9, `config_final.json`) debe corregirse en esa frase. **Alternativa si el asesor no acepta el cambio:** la opción A, con el resultado de la prueba reportado con el mismo peso y no como apéndice.

### 18.2 Resto de elecciones

| Tema | Elección propuesta | Razón |
|---|---|---|
| Margen δ de H1 | `[x]` **Contraste primario sin margen (δ = 0), como dice la H1 del PPI** («estadísticamente superior»). `[x]` Se reporta además, como criterio secundario de relevancia práctica, si la diferencia supera 0.05 de PR-AUC. | El 0.05 no tiene hoy una derivación empírica. Un margen sin justificación cambia la hipótesis, no solo la prueba. Reportar ambos conserva la hipótesis aprobada y da la lectura práctica. |
| Prueba pareada | `[x]` Wilcoxon de rangos con signo, con Holm contra G0 | PR-AUC por bloque no tiene por qué ser normal; el PPI es no paramétrico. Ya está implementada y probada. |
| Resumen de las 3 semillas por bloque | `[x]` Media | Una sola cifra por bloque y condición; la dispersión entre semillas se reporta aparte. |
| Criterio de H2 (≤ 10 ms, batch 1) | `[x]` **Mediana decide; p95 e IC 95 % se reportan** | El PPI usa la latencia mediana en la frontera de Pareto; el criterio y el análisis principal deben coincidir. |
| Modo de llamada en la medición de latencia | `[x]` **`tf.function` (grafo compilado) es el principal; eager se reporta como secundario** | El PPI pide un calentamiento para reducir el efecto de «compilación interna», lo que presupone un grafo compilado, y un despliegue real no usaría eager. En una prueba en el sandbox de Claude (no en su laptop) la diferencia fue de 10 a 40 veces, así que el veredicto de H2 depende de esta elección y debe fijarse antes de medir. |
| Fuga del umbral | `[x]` Documentar como limitación; no cambiar el bloque del 80 % | El umbral solo afecta F1 y FPR al recall 0.90, no el PR-AUC primario. Reservar otro bloque obliga a recalcular el presupuesto. |

### 18.3 Análisis complementario de H3 bajo el Esquema A
`[x]` Aprobado: τ y ρ con PR-AUC **agregado por pliegue**, como sensibilidad y sin sustituir el análisis principal. Con 71, 102 y 83 bloques evaluables por pliegue y pocos positivos, es probable que el IC de τ cruce 0.7 y el resultado sea no concluyente; esto se fija ahora, antes de ver resultados.

### 18.4 Registro
Aprobado el 3 de noviembre de 2026 (ver acta de la reunión). Asesor informado el: ______. Después: corregir el PPI (§1.6, §2.9) y `config_final.json` en lo que cambie 18.1, y poner el short paper de acuerdo.

---

## Decisión 19 — Entorno de entrenamiento y fe de erratas de la Decisión 16

### 19.1 Entorno
- Se entrena y se mide la latencia en **CPU, Windows**, con `requirements-lock.txt` (TensorFlow 2.16.2), lote 64 y las mismas semillas. La GPU bajo WSL2 **no se usa** para resultados (fue más lenta que la CPU con estos modelos).
- Los tiempos de entrenamiento (VD6) solo se comparan entre arquitecturas si se midieron en la misma máquina. `configs/hardware.json` ya está actualizado.

### 19.2 Fe de erratas de la Decisión 16 (el texto original se conserva)
- **Dice:** «unas 51 horas por arquitectura y esquema». **Es:** el total de las 12 combinaciones (77–89 días del plan literal ÷ 36 = 51–59 h). Y los extras de esa decisión (~102 h, ~14 h, ~34 h) se calcularon con 30 épocas, no con las 10 del reentrenamiento final.
- **Medición** (`semana9_cronometro_fit.py`, CPU Windows, pliegue 3, Esquema B, lote 64): `fit()` en ms por paso: autoencoder 2.55; CNN-1D 4.43; LSTM 9.95; CNN-LSTM-GRU 10.68; GRU 12.85; Transformer 19.55.
- **Presupuesto** (`semana9_presupuesto.py`, `results/eda/presupuesto_computo.md`): **185 h, unos 7.7 días continuos**, con el bloque del 80 % a 1.333 veces el pliegue 3. No incluye validación por época, puntuación final ni G7. Quedan 648 h por máquina hasta el 29 de noviembre.

### 19.3 Reducción de la búsqueda
`[x]` **Mantener la reducción tal como está escrita** (1 semilla, hasta 10 épocas, 1 de cada 4 ventanas, 12 configuraciones). **Razón:** está declarada antes de los resultados, el tiempo de las semanas 10 y 11 es justo y falta medir el costo de la validación por época. Relajarla solo se justifica si la primera corrida real muestra que esa validación cuesta poco, y debe decidirse **antes** de ver resultados de las redes, no después. (Referencia: pasar la búsqueda a 3 semillas subiría el total a unas 370 h.)

### 19.4 Registro
Aprobado el 3 de noviembre de 2026 (ver acta de la reunión).

---

## Antes de sellar las Decisiones 15–17 (lo que sigo sin poder decidir yo)
1. **Confirmar la métrica de selección** de la Decisión 16 contra el texto del §2.7.1; el propio borrador de esa decisión dice «(Confirmar…)» y sigue abierto.
2. **Aplicar la errata** de 19.2 antes de sellar, para no sellar una cifra equivocada.
3. **«144 frente a 132 corridas»:** el Cuadro 8 y la Figura 2 del PPI dan cifras distintas. No puedo deducir cuál es la correcta sin la definición de «corrida» del equipo. Deben fijarla y corregir la otra.
4. **G7:** smoke-test el 6 de noviembre con un límite de **un día laborable**; si no produce puntuaciones sobre las ventanas en ese plazo, se activa la contingencia de 7 condiciones, documentando el intento. El plazo y el responsable los confirma el equipo.
