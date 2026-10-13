"""
Actualiza README.md: reemplaza la seccion '## Estructura del repositorio' y la linea
'**Estado actual:**'. Hace copia de seguridad en README.md.bak. Idempotente.
Uso:  python actualizar_readme.py
"""
import re
import shutil
from pathlib import Path

ESTRUCTURA = '''## Estructura del repositorio

```
.
├── configs/                              (criterios versionados)
│   ├── config_preprocesamiento.json      (exclusion de sensores, hora pico)
│   ├── particion_temporal.json           (3 pliegues walk-forward + prueba final)
│   ├── normalizacion_zscore.json         (mu y sigma por pliegue + bloque final)
│   ├── config_esquema_A.json             (Esquema A causal, Decision 8)
│   ├── config_esquema_B.json             (Esquema B, Decisiones 9, 10 y 12)
│   ├── config_g0.json                    (G0: grilla y regla de seleccion, Decision 11)
│   ├── config_final.json                 (se sella en el gate de la semana 7)
│   └── hardware.json                     (caracterizacion del entorno)
├── data/                                 (NO versionado; se regenera)
├── results/eda/                          (reportes .json/.md versionados; ver LEEME_resultados.md)
├── docs/
│   ├── imagenes/                         (figuras del PPI)
│   ├── informes/                         (cierres semanales por integrante, PDF)
│   ├── paper/                            (figuras del articulo corto)
│   └── justificacion_dataset_pems04.pdf
├── scripts_paper/                        (generacion de figuras del articulo)
├── src/
│   ├── ventanas.py                       (ventaneo reutilizable)
│   ├── inyeccion_anomalias.py            (modulo del Esquema B)
│   └── g0_linea_base.py                  (G0: score causal y evaluacion por bloque)
├── eda_pems04.py                         (Semana 1)
├── semana2_preprocesamiento.py           (Semana 2)
├── semana3_particion.py / semana3_normalizacion.py / semana3_ventanas.py   (Semana 3)
├── semana4_esquema_A.py                  (Semana 4, vigente)
├── semana4_normalizacion_final.py        (Semana 4, vigente)
├── semana4_esquema_B.py                  (Semana 4, SUPERADO por semana5_esquema_B_por_pliegue.py)
├── semana4_comparacion_A_vs_B.py         (Semana 4, SUPERADO por semana5_comparacion_A_vs_B.py)
├── semana5_esquema_B_por_pliegue.py      (Semana 5)
├── semana5_g0.py                         (Semana 5)
├── semana5_validacion_pipeline.py        (Semana 5)
├── semana5_comparacion_A_vs_B.py         (Semana 5)
├── semana6_g0_ampliado.py                (Semana 6: G0 por bloque, recall 0.90, tipos, normalizacion)
├── auditar_repo.py                       (auditoria de lo versionado)
├── bitacora_decisiones_metodologicas.md  (Decisiones 1-14)
├── CHANGELOG.md, CITATION.cff, LICENSE, requirements.txt, requirements-lock.txt
└── README.md
```

**Orden de ejecucion:** `eda_pems04.py` → `semana2_preprocesamiento.py` → `semana3_particion.py` → `semana3_normalizacion.py` → `semana4_normalizacion_final.py` → `semana4_esquema_A.py` → `semana5_esquema_B_por_pliegue.py` → `semana5_g0.py` → `semana5_validacion_pipeline.py` → `semana5_comparacion_A_vs_B.py` → `semana6_g0_ampliado.py`.'''
ESTADO = '''**Estado actual:** Semana 6 de 12 en curso. Datos, Esquemas A y B (perfil por sensor), G0 (con salida por bloque) y 50 controles de fuga ejecutados y verificados; pendientes el codigo de G1-G7, la medicion de una epoca real y las correcciones del protocolo.'''

p = Path("README.md")
t = p.read_text(encoding="utf-8-sig")
shutil.copy(p, "README.md.bak")

# 1) seccion de estructura: desde su encabezado hasta el siguiente '## ' (o fin)
m = re.search(r"^## Estructura del repositorio.*?(?=^## |\Z)", t, flags=re.S | re.M)
if not m:
    raise SystemExit("No encontre '## Estructura del repositorio'. README.md no se modifico.")
t = t[:m.start()] + ESTRUCTURA.rstrip() + "\n\n" + t[m.end():]

# 2) linea de estado: reemplaza la existente o la agrega tras el titulo
lineas = t.split("\n")
nuevo = ESTADO.split("\n")[0]
idx = [k for k, l in enumerate(lineas) if l.startswith("**Estado actual:**")]
if idx:
    lineas[idx[0]] = nuevo
else:
    lineas.insert(1, "\n" + nuevo)
p.write_text("\n".join(lineas), encoding="utf-8")
print("README.md actualizado (copia en README.md.bak).")
for i, l in enumerate(t.splitlines(), 1):
    if ("semana4_esquema_B.py" in l or "semana4_comparacion_A_vs_B.py" in l) and "SUPERAD" not in l.upper():
        print(f"AVISO README linea {i}: menciona un script superado sin marcarlo -> edita a mano: {l.strip()[:100]}")
