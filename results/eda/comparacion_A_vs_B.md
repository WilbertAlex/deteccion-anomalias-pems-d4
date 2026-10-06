# Comparacion Esquemas A vs B (PE1)

## Prevalencia por particion

| Particion | A (%) | B (%) | Razon B/A |
|---|---|---|---|
| test | 0.0989 | 3.0549 | 30.90 |
| pliegue1_train | 0.1250 | 3.0527 | 24.43 |
| pliegue1_val | 0.0947 | 2.9880 | 31.57 |
| pliegue2_train | 0.1091 | 3.0204 | 27.67 |
| pliegue2_val | 0.0988 | 3.0241 | 30.61 |
| pliegue3_train | 0.1056 | 3.0216 | 28.61 |
| pliegue3_val | 0.0860 | 3.0562 | 35.52 |

## Solapamiento (Jaccard)

| Particion | Jaccard | A anomalos | B anomalos | En ambos |
|---|---|---|---|---|
| test | 0.0010 | 978 | 30216 | 31 |
| pliegue1_train | 0.0012 | 1131 | 27697 | 34 |
| pliegue1_val | 0.0004 | 936 | 29546 | 12 |
| pliegue2_train | 0.0008 | 2067 | 57243 | 46 |
| pliegue2_val | 0.0015 | 977 | 29903 | 45 |
| pliegue3_train | 0.0010 | 3044 | 87146 | 91 |
| pliegue3_val | 0.0006 | 851 | 30229 | 20 |

## Bloques con ambas clases

- Esquema A: 692 bloques
- Esquema B: 2037 bloques
- Total evaluables: 2037


## Análisis del solapamiento

Los coeficientes de Jaccard entre los Esquemas A y B son **extremadamente bajos** en todas las particiones evaluadas (rango: 0.0004–0.0015, mediana 0.0010). Esto significa que menos del 0.15% de las anomalías detectadas por un esquema coinciden con el otro.

**Interpretación:**

- A pesar de que ambos esquemas declaran anomalías sobre la misma serie, **las anomalías que cada uno marca son casi disjuntas**.
- El Esquema A (proxy causal `|z|>3` sobre ventana móvil) detecta desviaciones estadísticas respecto al patrón reciente de cada sensor.
- El Esquema B (inyección controlada estratificada) introduce perturbaciones artificiales de tres tipos (puntual, contextual, colectiva) que no siempre coinciden con los extremos estadísticos locales.
- La razón B/A de prevalencia es ~30x, lo que confirma que los esquemas operan a escalas de sensibilidad muy distintas.

**Implicación para H3:**

Este resultado constituye evidencia empírica preliminar de que **el ranking de arquitecturas puede diferir entre esquemas**, porque las anomalías que cada uno define son prácticamente independientes. El análisis formal de H3 (τ de Kendall con IC bootstrap) se ejecutará en la Semana 11 con los resultados de la batería completa.

**Implicación para el contraste estadístico:**

El número de bloques sensor × pliegue con ambas clases es **692 bajo el Esquema A** y **2037 bajo el Esquema B** (de 2037 totales). Esto implica:

- Bajo el Esquema B, prácticamente todos los bloques son evaluables (99.99%).
- Bajo el Esquema A, el 34% de los bloques tienen solo una clase (habitualmente todos normales), lo que reduce el poder estadístico del contraste principal.
- Esta asimetría es inherente a la diferencia de prevalencia (0.10% vs 3.03%) y debe considerarse en la interpretación de H1 y H3.