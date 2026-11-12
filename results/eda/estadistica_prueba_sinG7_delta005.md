# Analisis estadistico (salida de semana9_estadistica.py)

Parametros: `{"referencia": "G0", "conjunto": "prueba", "delta": 0.05, "alternativa": "greater", "alfa": 0.05, "B_bootstrap": 1000, "semilla": 42}`

## Esquema A
Bloques: 84 evaluables comunes de 291. Condiciones: G0, G1, G2, G3, G4, G5, G6.

Friedman: chi2 = 241.42, p = 2.79e-49, W = 0.479

| Condicion | n | dif. media | IC95 % | p | p Holm | rechaza H0 |
|---|---|---|---|---|---|---|
| G1 | 84 | +0.3327 | [+0.2732, +0.3962] | 8.74e-11 | 4.37e-10 | si |
| G2 | 84 | +0.3234 | [+0.2647, +0.3798] | 4.2e-11 | 2.52e-10 | si |
| G3 | 84 | +0.2646 | [+0.2072, +0.3273] | 7.99e-10 | 1.82e-09 | si |
| G4 | 84 | -0.0460 | [-0.0886, -0.0043] | 1 | 1 | no |
| G5 | 84 | +0.2876 | [+0.2275, +0.3400] | 6.06e-10 | 1.82e-09 | si |
| G6 | 84 | +0.3154 | [+0.2579, +0.3736] | 9.27e-11 | 4.37e-10 | si |

## Esquema B
Bloques: 291 evaluables comunes de 291. Condiciones: G0, G1, G2, G3, G4, G5, G6.

Friedman: chi2 = 1199.67, p = 5.63e-256, W = 0.687

| Condicion | n | dif. media | IC95 % | p | p Holm | rechaza H0 |
|---|---|---|---|---|---|---|
| G1 | 291 | +0.6516 | [+0.6419, +0.6606] | 9.02e-50 | 5.41e-49 | si |
| G2 | 291 | +0.6511 | [+0.6417, +0.6604] | 9.02e-50 | 5.41e-49 | si |
| G3 | 291 | +0.6496 | [+0.6409, +0.6587] | 9.02e-50 | 5.41e-49 | si |
| G4 | 291 | +0.1034 | [+0.0824, +0.1234] | 3.71e-07 | 3.71e-07 | si |
| G5 | 291 | +0.6517 | [+0.6427, +0.6609] | 9.02e-50 | 5.41e-49 | si |
| G6 | 291 | +0.6517 | [+0.6424, +0.6606] | 9.02e-50 | 5.41e-49 | si |

## H3 (estabilidad del ranking)
tau = 0.619, IC95 % [0.333, 0.905]; rho = 0.821. Lectura segun el PPI: no concluyente (IC cruza 0.7). Bloques comunes: 84.
Ranking A: G1 > G2 > G6 > G5 > G3 > G0 > G4
Ranking B: G1 > G5 > G6 > G2 > G3 > G4 > G0
