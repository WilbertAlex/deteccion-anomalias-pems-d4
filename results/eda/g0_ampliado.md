# G0 ampliado (Semana 6): bloques, recall 0.90, tipos de anomalia, normalizacion bajo B

No cambia la seleccion de G0 (Decision 11). Reproduce `g0_resultados.json` (asserts).

## Recall 0.90 (umbral fijado en la validacion del pliegue 3, aplicado a la prueba)

| Esquema | Config. | Umbral | Recall prueba | FPR prueba | k por F1 solo en pliegue 3 | k Decision 11 |
|---|---|---|---|---|---|---|
| A | w=22 (elegida) | 0.852 | 0.917 | 0.6562 | 3.0 | 3.0 |
| A | w=11 (sensibilidad) | 0.455 | 0.893 | 0.7776 | 3.0 | - |
| B | w=22 (elegida) | 0.918 | 0.896 | 0.5425 | 3.0 | 3.0 |
| B | w=11 (sensibilidad) | 0.558 | 0.900 | 0.6869 | 3.0 | - |

## Esquema B: instantes anomalos por tipo (prueba)

| Tipo | % de instantes |
|---|---|
| puntual | 14.6 |
| contextual | 10.7 |
| colectiva | 74.7 |

## Esquema B: PR-AUC de G0 por tipo de anomalia (prueba; positivos = ese tipo, negativos = instantes normales)

| Config. | Tipo | PR-AUC | Lift | Bloques |
|---|---|---|---|---|
| G0_w22 | puntual | 0.905 | 198.0 | 291 |
| G0_w22 | contextual | 0.399 | 118.7 | 291 |
| G0_w22 | colectiva | 0.131 | 5.7 | 291 |
| G0_w11_sensibilidad | puntual | 0.769 | 168.1 | 291 |
| G0_w11_sensibilidad | contextual | 0.311 | 92.6 | 291 |
| G0_w11_sensibilidad | colectiva | 0.083 | 3.6 | 291 |

## Normalizacion bajo el Esquema B (flujo): serie original vs serie inyectada del train

| Bloque | mu original | mu B | dif. % | sigma original | sigma B | dif. % |
|---|---|---|---|---|---|---|
| pliegue1 | 209.97 | 209.21 | -0.36 | 155.62 | 161.75 | +3.93 |
| pliegue2 | 210.40 | 209.73 | -0.32 | 155.89 | 162.20 | +4.05 |
| pliegue3 | 211.83 | 211.17 | -0.31 | 156.93 | 163.33 | +4.08 |
| final | 215.17 | 214.54 | -0.30 | 158.59 | 165.06 | +4.08 |