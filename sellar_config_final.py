"""
Sella configs/config_final.json: registra la fecha REAL del sellado, quien aprobo y el SHA-256 de los archivos que
fijan las reglas (configs/config_busqueda.json, configs/config_modelos.json y bitacora_decisiones_metodologicas.md).

Uso (desde la raiz del proyecto):
  python sellar_config_final.py --fecha 2026-11-03 --aprobado-por "texto que identifica a quien aprobo"
  python sellar_config_final.py --ver        # muestra el sello y verifica que los archivos no cambiaron

Reglas (todo-o-nada; si algo falla no se escribe nada):
  - la fecha no puede ser futura;
  - las Decisiones 15 a 19 deben aparecer UNA vez cada una en la bitacora, y no debe quedar ninguna marcada «PROPUESTA»;
  - no se puede sellar dos veces; antes de escribir deja una copia configs/config_final.json.bak;
  - los hashes se calculan con los saltos de linea normalizados a LF, para que den lo mismo en Windows y en Linux/git.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ARCHIVOS = ["configs/config_busqueda.json", "configs/config_modelos.json", "bitacora_decisiones_metodologicas.md"]
FINAL = Path("configs/config_final.json")
DECISIONES = ["15", "16", "17", "18", "19"]


def sha(p):
    b = Path(p).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def cargar():
    raw = FINAL.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), (b"\r\n" in raw)


ap = argparse.ArgumentParser()
ap.add_argument("--fecha")
ap.add_argument("--aprobado-por", dest="aprobado")
ap.add_argument("--ver", action="store_true")
a = ap.parse_args()

for p in [FINAL] + [Path(x) for x in ARCHIVOS]:
    if not p.exists():
        sys.exit(f"Falta {p}. Ejecuta desde la raiz del proyecto.")

cfg, crlf = cargar()

if a.ver:
    s = cfg.get("sellado")
    if not cfg.get("fecha_sellado") or not s:
        sys.exit("config_final.json NO esta sellado.")
    print(f"Sellado el {cfg['fecha_sellado']}\nAprobado por: {s['aprobado_por']}\nDecisiones: {', '.join(s['decisiones'])}")
    ok = True
    for f, h in s["sha256"].items():
        h2 = sha(f)
        estado = "coincide" if h2 == h else "CAMBIO DESPUES DEL SELLO"
        ok &= h2 == h
        print(f"  {f}: {h[:16]}...  {estado}")
    sys.exit(0 if ok else 1)

if not a.fecha or not a.aprobado:
    sys.exit("Faltan --fecha y --aprobado-por (o usa --ver).")
if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.fecha):
    sys.exit("La fecha debe ser AAAA-MM-DD.")
if dt.date.fromisoformat(a.fecha) > dt.date.today():
    sys.exit("La fecha no puede ser futura.")
if cfg.get("fecha_sellado") or cfg.get("sellado"):
    sys.exit("config_final.json ya esta sellado. No se sella dos veces; una modificacion posterior es una desviacion y se registra en la bitacora.")

bit = Path("bitacora_decisiones_metodologicas.md").read_text(encoding="utf-8-sig")
for d in DECISIONES:
    n = len(re.findall(rf"^## Decisión {d} ", bit, flags=re.M))
    if n != 1:
        sys.exit(f"La Decision {d} aparece {n} veces en la bitacora (debe ser 1). No se sello nada.")
m = re.search(r"^\*\*.*PROPUESTA.*$", bit, flags=re.M)
if m:
    sys.exit("La bitacora aun tiene un encabezado marcado PROPUESTA:\n  " + m.group(0)[:200] + "\nApruebalo antes de sellar. No se sello nada.")
for f in ("configs/config_busqueda.json", "configs/config_modelos.json"):
    t = json.loads(Path(f).read_bytes().decode("utf-8-sig"))
    if "PROPUESTA" in str(t.get("estado", "")):
        sys.exit(f"{f} todavia dice PROPUESTA en su estado. No se sello nada.")

Path(str(FINAL) + ".bak").write_bytes(FINAL.read_bytes())
cfg["fecha_sellado"] = a.fecha
if "fecha_sellado_gate_semana_7" in cfg:
    cfg["fecha_sellado_gate_semana_7"] = a.fecha
cfg["estado"] = f"SELLADO el {a.fecha} (el gate de la semana 7 se traslado; ver Anexo C del PPI)"
cfg["sellado"] = {
    "aprobado_por": a.aprobado,
    "decisiones": DECISIONES,
    "sha256": {f: sha(f) for f in ARCHIVOS},
    "nota_hash": "SHA-256 con saltos de linea normalizados a LF. config_final.json no se incluye en el hash.",
}
s = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
FINAL.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
print(f"OK: config_final.json sellado el {a.fecha}.")
for f, h in cfg["sellado"]["sha256"].items():
    print(f"  {f}: {h}")
print("Siguiente: git add configs/config_final.json y commit. Verifica con: python sellar_config_final.py --ver")
