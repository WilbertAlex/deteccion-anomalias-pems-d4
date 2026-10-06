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