# Semana 5 (5 - 11 oct. 2026) - guia paso a paso

## Alcance completo de la semana (nada omitido)
| # | Tarea | Responsable | Estado |
|---|---|---|---|
| 1 | Corregir Esquema B: un arreglo por pliegue (Decision 10) | Choquechambi | Script listo, falta correr en tu PC |
| 2 | G0 con 12 configuraciones, seleccion solo en validacion (Decision 11) | Prieto | Script listo, falta correr |
| 3 | Validacion tecnica del pipeline (50 controles de fuga) | Mayta | Script listo, falta correr |
| 4 | Comparacion A vs B v2 (PE1) con B corregido | Mayta | Script listo, falta correr |
| 5 | Documentar hallazgos (bitacora, CHANGELOG, README, Anexo C) | Todos | Textos en esta guia |
| 6 | Corregir el error "34 % una sola clase" en Overleaf | Mayta | Texto en esta guia |
| 7 | Smoke-test de G7 (SI-DMSeq, venv aislado) | Choquechambi | PENDIENTE (requiere clonar el repo de los autores) |
| 8 | Correcciones del PPI (291 sensores, 132 corridas, H3 duplicada, Cuadros 3/6/8, Friedman, latencia) | Todos | PENDIENTE |
| 9 | Matriz de consistencia (Anexo B) con referencias a archivos | Mayta | PENDIENTE |
| 10 | Decision abierta: perfil contextual por sensor (PPI) vs agrupado (codigo actual) | Equipo | PENDIENTE de decidir |

## Texto de correccion para Overleaf (Anexo / seccion de resultados preliminares)
Reemplazar la frase "el 34 % de los bloques tienen una sola clase" por:
"solo el 29,2 % de los bloques de evaluacion (340 de 1 164) contienen ambas clases bajo el Esquema A, frente al 100 % bajo el Esquema B; el 70,8 % restante tiene una sola clase y no admite PR-AUC."
(Usar los numeros de comparacion_A_vs_B_v2 una vez corrida en tu PC.)

## CHANGELOG (agregar al inicio, antes de [0.4.0])
## [0.5.0] - Semana 5 (5-11 oct. 2026) - en curso
### Agregado
- `semana5_esquema_B_por_pliegue.py`: Esquema B por pliegue (Decision 10)
- `src/g0_linea_base.py` y `semana5_g0.py`: G0 con 12 configuraciones (Decision 11)
- `semana5_validacion_pipeline.py`: 50 controles de fuga temporal por perturbacion
- `semana5_comparacion_A_vs_B.py`: comparacion PE1 sobre bloques de evaluacion
### Corregido
- Esquema B: el arreglo unico sobrescribia pliegues y reutilizaba la misma semilla en train y val
- Texto del Anexo: fraccion de bloques con ambas clases bajo A (29 %, no 66 %)

## README - agregar a la estructura
src/g0_linea_base.py            (G0, semana 5)
semana5_esquema_B_por_pliegue.py, semana5_g0.py, semana5_validacion_pipeline.py, semana5_comparacion_A_vs_B.py
