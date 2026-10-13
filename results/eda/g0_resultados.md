# G0 - resultados (seleccion solo en validacion; prueba evaluada una vez)

PR-AUC = media sobre bloques (sensores con ambas clases). Lift = PR-AUC / prevalencia del bloque (azar = 1). **A: G0 comparte familia con el etiquetador; su resultado es techo del etiquetador, no evidencia de deteccion.**


## Esquema A

| Ventana | PR-AUC val | Lift val | Bloques val (p1,p2,p3) | F1 k=2.0 | F1 k=2.5 | F1 k=3.0 |
|---|---|---|---|---|---|---|
| 6 | 0.0200 | 6.5 | [71, 102, 83] | 0.012 | 0.012 | 0.011 |
| 11 | 0.0294 | 9.4 | [71, 102, 83] | 0.014 | 0.014 | 0.014 |
| 22 | 0.0426 | 13.7 | [71, 102, 83] | 0.017 | 0.020 | 0.020 |
| 44 | 0.0390 | 12.4 | [71, 102, 83] | 0.022 | 0.028 | 0.029 |

Seleccion: w*=22, k*=3.0

Prueba: PR-AUC **0.0838** (prevalencia 0.0034, lift 24.5x), F1 0.021, AUC-ROC 0.708, FPR 0.0322, bloques evaluables 84

## Esquema B

| Ventana | PR-AUC val | Lift val | Bloques val (p1,p2,p3) | F1 k=2.0 | F1 k=2.5 | F1 k=3.0 |
|---|---|---|---|---|---|---|
| 6 | 0.2444 | 8.0 | [291, 291, 291] | 0.121 | 0.143 | 0.188 |
| 11 | 0.2926 | 9.6 | [291, 291, 291] | 0.150 | 0.206 | 0.266 |
| 22 | 0.3445 | 11.3 | [291, 291, 291] | 0.182 | 0.254 | 0.320 |
| 44 | 0.3142 | 10.3 | [291, 291, 291] | 0.191 | 0.241 | 0.283 |

Seleccion: w*=22, k*=3.0

Prueba: PR-AUC **0.3453** (prevalencia 0.0306, lift 11.3x), F1 0.323, AUC-ROC 0.792, FPR 0.0269, bloques evaluables 291