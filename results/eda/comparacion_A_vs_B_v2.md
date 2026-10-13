# Comparacion Esquemas A vs B (v2, B por pliegue)

| Split | Pasos validos | Anom. A | Anom. B | A (%) | B (%) | En ambos | Jaccard | Bloques ambas clases A | Bloques ambas clases B |
|---|---|---|---|---|---|---|---|---|---|
| val_pliegue1 | 988,818 | 936 | 30,289 | 0.095 | 3.063 | 8 | 0.00026 | 71 | 291 |
| val_pliegue2 | 988,818 | 977 | 30,236 | 0.099 | 3.058 | 13 | 0.00042 | 102 | 291 |
| val_pliegue3 | 989,109 | 851 | 30,234 | 0.086 | 3.057 | 14 | 0.00045 | 83 | 291 |
| prueba | 989,109 | 978 | 30,237 | 0.099 | 3.057 | 38 | 0.00122 | 84 | 291 |

**Bloques con ambas clases**: A 340 de 1164 (29.2%); B 1164 de 1164 (100.0%).

Implicacion: bajo A, la fraccion indicada de bloques es evaluable para PR-AUC; el contraste de Friedman por bloques sensor x pliegue tendra menos bloques bajo A que bajo B.

## Anomalias por hora del dia (suma de los 4 splits)
| Hora | A | B |
|---|---|---|
| 00 | 79 | 5129 |
| 01 | 35 | 4665 |
| 02 | 10 | 5297 |
| 03 | 14 | 5024 |
| 04 | 185 | 5001 |
| 05 | 571 | 5005 |
| 06 | 667 | 4969 |
| 07 | 802 | 5038 |
| 08 | 354 | 5057 |
| 09 | 131 | 5189 |
| 10 | 59 | 4995 |
| 11 | 33 | 4806 |
| 12 | 29 | 4721 |
| 13 | 24 | 4916 |
| 14 | 233 | 5118 |
| 15 | 260 | 5230 |
| 16 | 182 | 5073 |
| 17 | 68 | 5218 |
| 18 | 3 | 5137 |
| 19 | 0 | 5016 |
| 20 | 0 | 5065 |
| 21 | 1 | 5130 |
| 22 | 1 | 5245 |
| 23 | 1 | 4952 |