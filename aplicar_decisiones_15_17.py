"""
Reemplaza en la bitacora las Decisiones 15-17 por la version vigente de bitacora_decisiones_15_16.md.
Sirve tanto si la bitacora no las tiene como si tiene una version anterior (sin D17 o sin la ablacion).
Idempotente. Hace copia .bak. Requiere que la Decision 14 ya este en la bitacora.
"""
import re, shutil
from pathlib import Path
B = Path("bitacora_decisiones_metodologicas.md"); N = Path("bitacora_decisiones_15_16.md")
txt = B.read_text(encoding="utf-8-sig"); nuevo = N.read_text(encoding="utf-8-sig").lstrip("\n")
assert "## Decisión 14" in txt, "falta la Decision 14: aplica primero aplicar_decisiones_13_14.py"
i = txt.find("## Decisión 15")
pre = txt[:i] if i != -1 else txt
pre = pre.rstrip()
if pre.endswith("---"):
    pre = pre[:-3].rstrip()
final = pre + "\n\n" + nuevo.rstrip() + "\n"
if final == txt:
    print("Sin cambios: la bitacora ya tiene la version vigente."); raise SystemExit
shutil.copy(B, str(B) + ".bak")
B.write_text(final, encoding="utf-8")
chk = B.read_text(encoding="utf-8")
for d in ("15", "16", "17"):
    assert len(re.findall(rf"^## Decisión {d} ", chk, flags=re.M)) == 1, f"Decision {d} no quedo exactamente una vez"
assert "Ablación de canales" in chk and "Reentrenamiento final sin conjunto independiente" in chk
print("Bitacora actualizada: Decisiones 15, 16 y 17 presentes una vez cada una (copia en bitacora_decisiones_metodologicas.md.bak).")
