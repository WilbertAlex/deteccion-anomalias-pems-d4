
---

## Decisión 15 — H1 bajo el Esquema B multicanal y ponderación de clases (fijada antes de entrenar las redes)
**Semana 8 (26 oct. – 1 nov. 2026) — PROPUESTA, vigente al ser aprobada por el equipo**

**Contexto:** la inyección del Esquema B altera flujo, ocupación y velocidad con multiplicadores fijos (Decisiones 10 y 12), mientras G0 opera solo sobre flujo. Por instantes, el 74.7 % de las anomalías son colectivas, 14.6 % puntuales y 10.7 % contextuales. En las pruebas de humo (`semana7_busqueda.py --prueba`, modelos diminutos, 1 época, 1/40 de las ventanas, reproducidas en el equipo) un LSTM de 4 unidades alcanzó PR-AUC medio por bloque de 0.52 y 0.60 en la validación del pliegue 1 del Esquema B. **No son resultados**, pero indican que una red puede aprender la firma multicanal de la inyección con muy poca capacidad. Una ventaja de las redes sobre G0 en el Esquema B podría reflejar esa firma y no capacidad de detectar anomalías reales.

**Decisión:**
1. H1 se contrasta como está definida (PR-AUC por bloque, Friedman/Holm), y **se reporta además el PR-AUC por tipo de anomalía** (puntual, contextual, colectiva) para todas las condiciones, con la composición por instantes.
2. Toda conclusión sobre H1 bajo el Esquema B se declara **condicionada a la firma multicanal** de la inyección; la lectura bajo el Esquema A (sin inyección) se presenta con el mismo peso.
3. **Ponderación de clases:** la corrida principal no usa pesos en ningún esquema. Se agrega un **análisis de sensibilidad** con 1 semilla y pesos `n_neg / n_pos` calculados en el entrenamiento de cada pliegue, para el Esquema A (prevalencia 0.10 %) y el B (3 %). Se reporta sin reemplazar la corrida principal.
4. La regla se fija antes de entrenar y no se modifica en función de los resultados.

**Limitación:** el análisis por tipo no elimina la firma multicanal; solo la hace visible.

## Decisión 16 — Reducción declarada de la búsqueda de hiperparámetros (§2.7.1)
**Semana 8 (26 oct. – 1 nov. 2026) — PROPUESTA, vigente al ser aprobada por el equipo**

**Contexto (medición del 27 oct., `results/eda/cronometro_epoca_cpu-windows.json`):** pliegue 3 del Esquema B = 2 963 253 ventanas de entrenamiento = 46 301 pasos por época con lote 64. En CPU con TensorFlow 2.16.2 (sin GPU), con un LSTM de medición (no es la configuración del Cuadro 6): 5.5–6.4 ms por paso, 4.3–5.0 min por época. El diseño literal del §2.7.1 son 12 configuraciones × 3 pliegues × 3 semillas × 6 arquitecturas × 2 esquemas = **1 296 entrenamientos** más 36 reentrenamientos finales; con 30 épocas por entrenamiento, 77 a 89 días de cómputo continuo para la búsqueda.

**Decisión** (parámetros en `configs/config_busqueda.json`):
- 12 configuraciones por arquitectura y esquema, muestreadas con semilla fija (`semilla_muestreo`).
- **1 semilla (42)** durante la búsqueda; hasta **10 épocas** con parada temprana (paciencia 3) por PR-AUC de validación.
- **Submuestreo fijo de ventanas** en entrenamiento y parada temprana: solo instantes `t` múltiplos de 4 (determinista, sin azar).
- Métrica de selección: media sobre los 3 pliegues de la media de PR-AUC por bloque en validación, solo bloques con ambas clases. *(Confirmar contra el texto del §2.7.1.)*
- La configuración ganadora de cada arquitectura y esquema se **reentrena con todas las ventanas y 3 semillas (42, 43, 44)** sobre el bloque del 80 %, y es lo único que entra al contraste final.
- G0: grilla completa de 12 (Decisión 11). G7: configuración por defecto publicada.

**Estimación con esta reducción** (misma medición): unas 51 horas por arquitectura y esquema para el LSTM de medición. Si el entrenamiento se pasa a GPU y se mide una aceleración, la reducción puede relajarse; cualquier relajación se decide **antes** de ver resultados de las redes y queda registrada en una decisión posterior.

**Limitaciones declaradas:** con 1 semilla, la selección de configuración lleva ruido de semilla; con submuestreo, la selección usa menos datos que el entrenamiento final; las ventanas consecutivas comparten 11 de 12 pasos, lo que reduce, pero no elimina, la pérdida de información.
