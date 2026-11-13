# Analisis estadistico (salida de semana9_estadistica.py)

Parametros: `{"referencia": "G0", "conjunto": "prueba", "delta": 0.0, "alternativa": "greater", "alfa": 0.05, "B_bootstrap": 1000, "semilla": 42}`

## Esquema A
Bloques: 0 evaluables comunes de 0. Condiciones: .

referencia G0 ausente o sin bloques evaluables comunes

## Esquema B
Bloques: 291 evaluables comunes de 291. Condiciones: G0, G1-abl, G2-abl, G3-abl, G5-abl, G6-abl.

Friedman: chi2 = 1132.48, p = 1.23e-242, W = 0.778

| Condicion | n | dif. media | IC95 % | p | p Holm | rechaza H0 |
|---|---|---|---|---|---|---|
| G1-abl | 291 | +0.5340 | [+0.5220, +0.5453] | 9.02e-50 | 4.51e-49 | si |
| G2-abl | 291 | +0.4980 | [+0.4833, +0.5117] | 9.02e-50 | 4.51e-49 | si |
| G3-abl | 291 | +0.5942 | [+0.5845, +0.6030] | 9.02e-50 | 4.51e-49 | si |
| G5-abl | 291 | +0.6026 | [+0.5940, +0.6115] | 9.02e-50 | 4.51e-49 | si |
| G6-abl | 291 | +0.6099 | [+0.6012, +0.6195] | 9.02e-50 | 4.51e-49 | si |

## H3 (estabilidad del ranking)
No aplicable: 0 condiciones y 0 bloques comunes
