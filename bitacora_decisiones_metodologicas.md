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

- **Estado: cerrada.**

---

## Próximas decisiones pendientes (Semana 4, 28 sep. – 4 oct.)
- [ ] Parámetros exactos del Esquema A: tamaño de ventana histórica y umbral de desviación
- [ ] Semillas y parámetros del generador de anomalías del Esquema B
