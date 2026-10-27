"""
Auditoria del repositorio: que existe, que esta versionado y que falta.
Uso (desde la carpeta del proyecto):  python auditar_repo.py
Solo LEE; no modifica nada. Termina con la lista de pendientes.
"""
import subprocess
from pathlib import Path


def git(*a):
    r = subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip()


tracked = set(git("ls-files").splitlines())
pend = []


def ok(msg): print(f"  [OK ] {msg}")
def falta(msg): print(f"  [FALTA] {msg}"); pend.append(msg)


def txt(p):
    try:
        return Path(p).read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


ESPERADOS = {
    "Configuracion": ["configs/config_esquema_A.json", "configs/config_esquema_B.json", "configs/config_final.json",
                      "configs/config_g0.json", "configs/config_preprocesamiento.json", "configs/hardware.json",
                      "configs/normalizacion_zscore.json", "configs/particion_temporal.json"],
    "Codigo (raiz)": ["eda_pems04.py", "semana2_preprocesamiento.py", "semana3_particion.py", "semana3_normalizacion.py",
                      "semana3_ventanas.py", "semana4_esquema_A.py", "semana4_esquema_B.py", "semana4_normalizacion_final.py",
                      "semana4_comparacion_A_vs_B.py", "semana5_esquema_B_por_pliegue.py", "semana5_g0.py",
                      "semana5_validacion_pipeline.py", "semana5_comparacion_A_vs_B.py", "semana6_g0_ampliado.py",
                      "semana6_verificar_datos.py", "semana6_cronometro_epoca.py"],
    "Codigo (src)": ["src/ventanas.py", "src/inyeccion_anomalias.py", "src/g0_linea_base.py", "src/datos.py", "src/entrenar.py"],
    "Documentos raiz": ["README.md", "CHANGELOG.md", "CITATION.cff", "LICENSE", "requirements.txt", ".gitignore",
                        "bitacora_decisiones_metodologicas.md", "requirements-lock.txt"],
    "Informes": ["docs/informes/informe_avance_por_integrante.pdf"] +
                [f"docs/informes/semana{i}_por_integrante.pdf" for i in range(1, 6)] +
                ["docs/justificacion_dataset_pems04.pdf"],
    "Imagenes": ["docs/imagenes/arquitectura_sistema.png", "docs/imagenes/cronograma_ppi_fechas.png",
                 "docs/imagenes/flujo_datos_modelos.png", "docs/imagenes/pipeline_estado_detallado.png"],
    "Evidencia (results/eda)": ["results/eda/prevalencia_esquema_A.json", "results/eda/prevalencia_esquema_B_por_pliegue.json",
                                "results/eda/comparacion_A_vs_B_v2.json", "results/eda/comparacion_A_vs_B_v2.md",
                                "results/eda/g0_resultados.json", "results/eda/g0_resultados.md",
                                "results/eda/validacion_pipeline.json", "results/eda/sensores_excluidos.csv",
                                "results/eda/g0_por_bloque.csv", "results/eda/verificacion_datos.json", "results/eda/g0_ampliado.json", "results/eda/g0_ampliado.md",
                                "results/eda/LEEME_resultados.md"],
    "CI": [".github/workflows/verificar-entorno.yml"],
}
print("1) Archivos esperados: existen y estan versionados")
for grupo, lista in ESPERADOS.items():
    print(f" - {grupo}")
    for f in lista:
        if not Path(f).exists():
            falta(f"{f} no existe")
        elif f not in tracked:
            falta(f"{f} existe pero NO esta versionado (git add)")
        else:
            ok(f)

print("\n2) No debe estar versionado (datos, entorno)")
malos = [f for f in tracked if f.startswith(("data/", "venv/", "venv_g7/")) and not f.endswith(".gitkeep")]
for m in malos: falta(f"versionado indebidamente: {m}")
if not malos: ok("ningun dato ni entorno virtual en git")
gi = txt(".gitignore")
for pat in ("venv", "data/raw", "data/processed", "venv_g7"):
    (ok if pat in gi else falta)(f".gitignore cubre {pat}")

print("\n3) Documentacion al dia")
bit = txt("bitacora_decisiones_metodologicas.md")
for n in ("Decisión 8", "Decisión 9", "Decisión 10", "Decisión 11", "Decisión 12", "Decisión 13", "Decisión 14"):
    (ok if n in bit else falta)(f"bitacora contiene '{n}'")
ch = txt("CHANGELOG.md")
(ok if "0.5.0" in ch else falta)("CHANGELOG tiene la entrada [0.5.0] (Semana 5)")
rd = txt("README.md")
for k in ("semana5_g0.py", "g0_linea_base.py", "semana5_esquema_B_por_pliegue.py", "config_g0.json", "semana6_g0_ampliado.py"):
    (ok if k in rd else falta)(f"README menciona {k}")
(ok if "Semana 6" in rd else falta)("README indica el estado de la Semana 6")
# H-09: el README no debe mandar a ejecutar scripts superados fuera de su anotacion de "SUPERADO"
for i, l in enumerate(rd.splitlines(), 1):
    if ("semana4_esquema_B.py" in l or "semana4_comparacion_A_vs_B.py" in l) and "SUPERAD" not in l.upper():
        falta(f"README linea {i} menciona un script superado sin marcarlo: {l.strip()[:90]}")
lock = txt("requirements-lock.txt")
for pat in ("file:///", "Downloads", "@ file"):
    if pat in lock:
        falta(f"requirements-lock.txt contiene una ruta local ({pat}): no es reproducible por otras personas")
(ok if "torch" not in lock.lower() else falta)("requirements-lock.txt sin torch (va en venv_g7)")
ci = txt(".github/workflows/verificar-entorno.yml")
(ok if ("compileall" in ci or "py_compile" in ci) else falta)("CI compila el codigo")
(ok if "src" in ci else falta)("CI cubre src/")

print("\n4) Archivos de la version anterior del Esquema B (superados por la Decision 10)")
for f in ("data/processed/features_esquema_B.npz", "data/processed/labels_esquema_B.npz"):
    if Path(f).exists():
        falta(f"{f} existe: es el Esquema B v1 (un solo arreglo); borrarlo o renombrarlo para no usarlo por error")
    else:
        ok(f"{f} ya no existe")
leeme = txt("results/eda/LEEME_resultados.md")
(ok if "superad" in leeme.lower() else falta)("results/eda/LEEME_resultados.md marca los resultados v1 como superados")

print("\n5) Estado de git")
print(git("status", "-sb").splitlines()[0] if git("status", "-sb") else "(sin repo)")
if "ahead" in git("status", "-sb").splitlines()[0]:
    falta("hay commits locales sin subir: git push")
sin = [l for l in git("status", "--porcelain").splitlines() if l.strip()]
for l in sin: falta(f"cambio sin commit: {l}")
if not sin: ok("arbol de trabajo limpio")

print(f"\n{'=' * 60}\nPENDIENTES: {len(pend)}")
for p in pend: print(" *", p)
