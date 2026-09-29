# Bitácora de decisiones metodológicas — PPI Detección de anomalías PeMS-D4

Este documento registra cada decisión de diseño tomada durante el preprocesamiento,
con su justificación y evidencia. Alimenta directamente la Sección 2.3 del protocolo
(Variables de entrada y preprocesamiento) y la Sección 2.11 (Reproducibilidad).

---

## Decisión 1 — Fuente e integridad del dataset
**Semana 1 (7–13 sep. 2026)**

- Fuente: `github.com/Davidham3/ASTGCN/blob/master/data/PEMS04/pems04.npz`
- Hash SHA-256 verificado (dos veces, entornos independientes):
  `95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a`
- Forma confirmada: `(16992, 307, 3)` — coincide con lo especificado en el Cuadro 4 del protocolo.
- **Estado: cerrada.**

---

## Decisión 2 — Valores faltantes: no existen como NaN en el archivo crudo
**Semana 1–2**

- Verificación directa: `np.isnan(raw).sum() == 0` y `np.isinf(raw).sum() == 0` sobre
  la totalidad del arreglo.
- **Implicación:** el paso 2 del protocolo ("identificación de valores faltantes y
  tratamiento mediante interpolación temporal") no aplica en el sentido literal de
  valores NaN, porque el archivo ya no los contiene (posible imputación previa por
  parte de Song et al. al construir PEMS04, no documentada por ellos).
- El problema real de calidad detectado no son NaN, sino **valores congelados o en
  cero de forma sostenida**, que se trata como problema de calidad de sensor
  (Decisión 3), no como hueco a interpolar.
- **Estado: cerrada.**

---

## Decisión 3 — Exclusión de sensores de baja calidad
**Semana 1–2**

Se detectaron dos patrones de falla de sensor mediante EDA:
- 24 sensores con más del 5% de observaciones de flujo en cero (9 de ellos, más del 20%).
- 58 sensores con al menos una racha de 2+ horas de flujo exactamente congelado
  (mismo valor sin variación).

**Criterio de exclusión adoptado** (documentado y fijo para todo el pipeline):
> Se excluye una estación si cumple **al menos una** de las siguientes condiciones:
> 1. Más del 20% de sus observaciones de flujo son exactamente cero.
> 2. Presenta al menos una racha de 24 horas consecutivas (288 pasos) con el
>    mismo valor de flujo exacto.

**Resultado de aplicar el criterio:**
- Sensores excluidos: **16 de 307 (5.2%)**
- Sensores retenidos: **291 de 307 (94.8%)**

**Justificación del umbral de 24h** (en vez del umbral inicial de exploración de 2h):
un umbral de 2h excluiría 58 sensores (~19% del dataset), una pérdida excesiva para
un problema que en la mayoría de esos casos puede ser tráfico real de madrugada.
El umbral de 24h aísla los casos donde no cabe otra explicación razonable que una
falla de sensor, manteniendo una pérdida de datos mínima y defendible.

**Por qué exclusión y no interpolación:** interpolar una racha de 24–38 horas
significaría inventar casi un día completo de tráfico sintético dentro de una serie
que se presenta como real — contradice el propósito del estudio. La exclusión
documentada es la decisión metodológicamente honesta.

- **Estado: cerrada.** Ver `config_preprocesamiento.json` para el criterio en
  formato máquina, y `results/eda/sensores_excluidos.csv` para el listado completo.

---

## Decisión 4 — Regla de hora pico
**Semana 2 (14–20 sep. 2026)**

**Método:** se calculó el perfil horario promedio de flujo (agregando los 291
sensores retenidos), y se definió como "hora pico" toda hora cuyo flujo promedio
supere el percentil 75 del perfil de 24 horas.

**Resultado empírico:**
```
Hora  Flujo promedio
15    326.9   <- máximo
16    325.7
14    322.5
17    320.6
13    303.7
12    292.5   <- percentil 75 = 292.3
```
**Regla adoptada:** `is_peak_hour = 1` si la hora está en `{12, 13, 14, 15, 16, 17}`,
`0` en cualquier otro caso. Regla fija, aplicada de forma idéntica a los 291
sensores, los 3 pliegues y el conjunto de prueba.

**Limitación reconocida y documentada (no oculta):** el perfil resultante es una
meseta amplia de mediodía-tarde, no el patrón bimodal clásico de hora punta
mañana/tarde. La causa más probable es que el promedio mezcla sensores de ambos
sentidos de circulación (uno pica en la mañana, el otro en la tarde), y al
promediarlos el patrón se aplana. Se documenta esta limitación en la Sección 2.3
del protocolo en vez de ocultarla; una regla por sentido de circulación queda
como posible refinamiento futuro, no incluido en el alcance actual por razones
de cronograma.

- **Estado: cerrada.**

---

## Próximas decisiones pendientes (Semana 3)
- [ ] Parámetros exactos de la partición walk-forward (tamaño de cada pliegue)
- [ ] Verificación de que la normalización z-score no usa información de validación/prueba
