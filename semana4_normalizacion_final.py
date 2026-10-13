"""
Semana 4 - Normalizacion del bloque 80% completo (para reentrenamiento final).

El PPI (Seccion 2.7.1, paso 3) indica que el reentrenamiento final de la
configuracion ganadora se hace sobre el bloque 80% inicial completo, con
normalizacion ajustada UNICAMENTE con ese bloque.

Este script calcula mu y sigma del bloque [0, 13593) para flow, occupancy
y speed, y los anade como entrada "final" en normalizacion_zscore.json.

Requiere: data/processed/pems04_semana2.npz
          configs/particion_temporal.json
          configs/normalizacion_zscore.json
Genera:   configs/normalizacion_zscore.json (actualizado con entrada "final")
"""
import json
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/pems04_semana2.npz")
PARTICION_PATH = Path("configs/particion_temporal.json")
NORM_PATH = Path("configs/normalizacion_zscore.json")

d = np.load(DATA_PATH)
raw = d["data"]  # (16992, 291, 3)
nombres_var = ["flow", "occupancy", "speed"]

with open(PARTICION_PATH, "r", encoding="utf-8") as f:
    particion = json.load(f)

with open(NORM_PATH, "r", encoding="utf-8") as f:
    norm = json.load(f)

test_start = particion["test_idx"][0]
print(f"Bloque 80% inicial: pasos [0, {test_start}) = {test_start} pasos")

bloque = raw[:test_start, :, :]
print(f"Forma del bloque: {bloque.shape}")

final_stats = {
    "rango_pasos": [0, test_start],
    "variables": {},
    "fecha_calculo": "2026-10-06",
}

print("\nEstadisticas del bloque 80% completo:")
for i, nombre in enumerate(nombres_var):
    col = bloque[:, :, i]
    mu = float(col.mean())
    sigma = float(col.std())
    assert sigma > 0, f"Variable {nombre}: sigma = 0 - revisar"
    final_stats["variables"][nombre] = {"mean": mu, "std": sigma}
    print(f"  {nombre:10s} -> media={mu:8.4f}  desv.std={sigma:8.4f}")

media_p3 = norm["pliegues"][2]["variables"]["flow"]["mean"]
media_final = final_stats["variables"]["flow"]["mean"]
assert media_p3 != media_final, "La media final no debe ser identica a la del pliegue 3"
print(f"\nVerificacion: media flow pliegue 3 = {media_p3:.4f}, "
      f"media flow final = {media_final:.4f} (difieren correctamente)")

norm["final"] = final_stats

with open(NORM_PATH, "w", encoding="utf-8") as f:
    json.dump(norm, f, indent=2, ensure_ascii=False)

print(f"\nNormalizacion final agregada a {NORM_PATH}")
print("Semana 4 - Normalizacion del bloque 80% completa.")