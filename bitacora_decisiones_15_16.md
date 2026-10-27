
---

## Decisión 15 — H1 bajo el Esquema B multicanal y ponderación de clases (fijada antes de entrenar las redes)
**Semana 8 (26 oct. – 1 nov. 2026) — PROPUESTA, vigente al ser aprobada por el equipo**

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
**Semana 8 (26 oct. – 1 nov. 2026) — PROPUESTA, vigente al ser aprobada por el equipo**

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

## Decisión 17 — Especificación de arquitecturas (Cuadro 6) y espacio de búsqueda
**Semana 8 (26 oct. – 1 nov. 2026) — PROPUESTA, vigente al ser aprobada por el equipo**

**Contexto:** el Cuadro 6 del PPI fija la configuración inicial de G1–G6 pero no todos los detalles de implementación, y el PPI exige registrar los rangos de la búsqueda en archivos de configuración sin enumerarlos. *(La especificación se tomó de la versión del PPI del 1 de septiembre; confirmar que no cambió.)*

**Decisión:**
1. **Configuración inicial** (`configs/config_modelos.json`, clave `inicial`), verificada con `assert` sobre la estructura de cada modelo (`semana7_humo_modelos.py`, 54 controles): G1 y G2, dos capas de 64 y 32 unidades, Adam 1e-3, dropout 0.2; G3, Conv1D de 32, 64 y 128 filtros, kernel 3, GlobalMaxPooling1D, AdamW; G4, Flatten (96) → 32 → 16 → 8 → simétrico → 96; G5, 2 bloques, 4 cabezas, embedding 64, feed-forward 128, AdamW 5e-4, dropout 0.1; G6, Conv1D 64 (kernel 3), LSTM 64, GRU 32, Adam 1e-3, dropout 0.2. Lote de 64 y parada temprana por PR-AUC de validación (en G4, por pérdida de validación sobre ventanas normales).
2. **Interpretaciones no fijadas por el Cuadro 6:** el dropout de las capas recurrentes es el argumento `dropout` de la capa; convoluciones con padding causal; Transformer con posición aprendida y GlobalAveragePooling1D; **weight decay de AdamW = 1e-4** (G3 y G5; el Cuadro 6 dice "decaimiento de pesos" sin valor); regularización L2 del autoencoder = 0 en la configuración inicial.
3. **Espacio de búsqueda** (clave `espacio`): vecindad de la configuración inicial en anchura de capas, dropout, tasa de aprendizaje, kernel y decaimiento de pesos. La **configuración 0 es siempre la del Cuadro 6**; las 11 restantes se muestrean con semilla fija sin repetirla. El presupuesto es de 12 configuraciones por arquitectura, igual que G0.
4. G4 usa como score el error cuadrático medio de reconstrucción y se entrena solo con ventanas de etiqueta 0; el percentil del umbral que menciona el PPI solo afecta a F1 y FPR, no a PR-AUC.

**Limitación:** los rangos son una propuesta del equipo; el PPI solo exige que queden registrados antes de la ejecución.
