# Analisis estadistico (salida de semana9_estadistica.py)

Parametros: `{"referencia": "G0", "conjunto": "prueba", "delta": 0.0, "alternativa": "greater", "alfa": 0.05, "B_bootstrap": 1000, "semilla": 42}`

## Esquema A
Bloques: 0 evaluables comunes de 0. Condiciones: .

referencia G0 ausente o sin bloques evaluables comunes

## Esquema B
Bloques: 291 evaluables comunes de 291. Condiciones: C1-LR, C2-LR-flujo, C3-GBM, C4-GBM-flujo, G0.

Friedman: chi2 = 867.53, p = 1.81e-186, W = 0.745

| Condicion | n | dif. media | IC95 % | p | p Holm | rechaza H0 |
|---|---|---|---|---|---|---|
| C1-LR | 291 | +0.4750 | [+0.4566, +0.4934] | 1.23e-49 | 3.61e-49 | si |
| C2-LR-flujo | 291 | +0.0428 | [+0.0288, +0.0552] | 7.23e-09 | 7.23e-09 | si |
| C3-GBM | 291 | +0.4423 | [+0.4277, +0.4572] | 9.31e-50 | 3.61e-49 | si |
| C4-GBM-flujo | 291 | +0.4832 | [+0.4700, +0.4963] | 9.02e-50 | 3.61e-49 | si |

## H3 (estabilidad del ranking)
No aplicable: 0 condiciones y 0 bloques comunes
