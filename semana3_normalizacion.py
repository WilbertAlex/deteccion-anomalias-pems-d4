"""
Semana 3 - Tarea 2: Normalizacion z-score por pliegue.

Regla (Seccion 2.3 del protocolo):
    z = (x - mu_train) / sigma_train
mu_train y sigma_train se estiman UNICAMENTE con el rango de entrenamiento
de cada pliegue (definido en configs/particion_temporal.json), nunca con
datos de validacion o del conjunto de prueba final. Esto se repite para
cada uno de los 3 pliegues, ya que el rango de entrenamiento crece en cada uno.

Se normalizan las 3 variables crudas (flujo, ocupacion, velocidad). Las
variables temporales (seno/coseno, hora pico) no se normalizan: ya estan
acotadas por construccion.

Requiere: data/processed/pems04_semana2.npz
          configs/particion_temporal.json
Genera:   configs/normalizacion_zscore.json
"""
import json
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/pems04_semana2.npz")
PARTICION_PATH = Path("configs/particion_temporal.json")
OUT_PATH = Path("configs/normalizacion_zscore.json")

d = np.load(DATA_PATH)
raw = d["data"]  # (16992, 291, 3) -> [flujo, ocupacion, velocidad]
nombres_var = ["flow", "occupancy", "speed"]

with open(PARTICION_PATH, "r", encoding="utf-8") as f:
    particion = json.load(f)

resultados = []
for p in particion["pliegues"]:
    k = p["pliegue"]
    t_start, t_end = p["train_idx"]
    train_slice = raw[t_start:t_end, :, :]  # (pasos_train, 291, 3)

    stats_pliegue = {"pliegue": k, "train_idx": [t_start, t_end], "variables": {}}
    print(f"\nPliegue {k} (train: indices [{t_start}, {t_end}), {t_end - t_start} pasos):")
    for i, nombre in enumerate(nombres_var):
        col = train_slice[:, :, i]
        mu = float(col.mean())
        sigma = float(col.std())
        assert sigma > 0, f"Pliegue {k}, variable {nombre}: desviacion estandar es 0 - revisar"
        stats_pliegue["variables"][nombre] = {"mean": mu, "std": sigma}
        print(f"  {nombre:10s} -> media={mu:8.4f}  desv.std={sigma:8.4f}")

    resultados.append(stats_pliegue)

# ------------------------------------------------------------------
# Verificacion: la media/desv.std debe variar entre pliegues, porque
# el rango de entrenamiento crece cada vez (si fueran identicas en los
# 3 pliegues, algo esta mal - probablemente se uso todo el dataset)
# ------------------------------------------------------------------
medias_flow = [r["variables"]["flow"]["mean"] for r in resultados]
assert len(set(medias_flow)) == 3, \
    "Las medias de flujo son identicas entre pliegues - la particion no se esta respetando"
print("\nVerificacion: las medias difieren correctamente entre los 3 pliegues (cada uno usa su propio rango de train).")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"pliegues": resultados}, f, indent=2, ensure_ascii=False)

print(f"\nNormalizacion guardada en {OUT_PATH}")
print("Semana 3 - Tarea 2 completa.")
