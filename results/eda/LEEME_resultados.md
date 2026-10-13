# results/eda: guia de versiones

Los reportes de esta carpeta que son pequenos se versionan como evidencia. Algunos fueron SUPERADOS:

| Archivo | Estado | Motivo |
|---|---|---|
| `prevalencia_esquema_B.json` | **Superado (v1)** | Esquema B con un solo arreglo para los 3 pliegues y misma semilla en train y val (Decision 10). Reemplazado por `prevalencia_esquema_B_por_pliegue.json`. |
| `comparacion_A_vs_B.json` / `.md` | **Superado (v1)** | Calculado sobre el Esquema B v1. Reemplazado por `comparacion_A_vs_B_v2.*`. |
| `prevalencia_esquema_A.json` | Vigente | Esquema A causal (Decision 8). |
| `prevalencia_esquema_B_por_pliegue.json` | Vigente | Esquema B corregido, con perfil contextual por sensor (Decision 12). Las prevalencias coinciden con las de la Decision 10. |
| `comparacion_A_vs_B_v2.*` | Vigente | Bloques de evaluacion (validacion 1-3 y prueba). |
| `g0_resultados.*` | Vigente | G0, 12 configuraciones (Decision 11), sobre el Esquema B con perfil por sensor (Decision 12). Los valores del Esquema B con perfil agrupado (prueba PR-AUC 0,3617) estan superados (0,3453). |
| `validacion_pipeline.json` | Vigente | 50 controles de fuga temporal. |
| `sensores_excluidos.csv` | Vigente | Los 16 sensores excluidos (Semana 2). |

Los archivos v1 se conservan por trazabilidad (historial de decisiones); no usarlos para analisis.
