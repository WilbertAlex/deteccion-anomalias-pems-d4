"""
Semana 3 - Tarea 3: Verificacion de la construccion de ventanas temporales.

No se guardan las ventanas completas de los 3 pliegues (serian varios GB
para modelos que aun no existen). Este script verifica que la funcion
construir_ventanas() de src/ventanas.py es correcta, demostrandola sobre
el Pliegue 1, y deja evidencia reproducible de esa verificacion.

Requiere: data/processed/pems04_semana2.npz
          configs/particion_temporal.json
          configs/normalizacion_zscore.json
          src/ventanas.py
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("src")))
from ventanas import construir_ventanas

DATA_PATH = Path("data/processed/pems04_semana2.npz")
PARTICION_PATH = Path("configs/particion_temporal.json")
NORM_PATH = Path("configs/normalizacion_zscore.json")

d = np.load(DATA_PATH)
raw = d["data"]  # (16992, 291, 3)
hour_sin, hour_cos = d["hour_sin"], d["hour_cos"]
day_sin, day_cos = d["day_sin"], d["day_cos"]
is_peak = d["is_peak_hour"]
n_steps, n_sensores, _ = raw.shape

with open(PARTICION_PATH, "r", encoding="utf-8") as f:
    particion = json.load(f)
with open(NORM_PATH, "r", encoding="utf-8") as f:
    norm = json.load(f)

# ------------------------------------------------------------------
# Construir el arreglo F=8 normalizado con las stats del Pliegue 1
# ------------------------------------------------------------------
pliegue1 = norm["pliegues"][0]
raw_norm = np.empty_like(raw)
for i, nombre in enumerate(["flow", "occupancy", "speed"]):
    mu = pliegue1["variables"][nombre]["mean"]
    sigma = pliegue1["variables"][nombre]["std"]
    raw_norm[:, :, i] = (raw[:, :, i] - mu) / sigma

temporales = np.stack([
    np.broadcast_to(hour_sin[:, None], (n_steps, n_sensores)),
    np.broadcast_to(hour_cos[:, None], (n_steps, n_sensores)),
    np.broadcast_to(day_sin[:, None], (n_steps, n_sensores)),
    np.broadcast_to(day_cos[:, None], (n_steps, n_sensores)),
    np.broadcast_to(is_peak[:, None], (n_steps, n_sensores)),
], axis=-1)  # (16992, 291, 5)

features = np.concatenate([raw_norm, temporales], axis=-1)  # (16992, 291, 8)
print(f"Arreglo de features construido: {features.shape} (F=8, normalizado con stats del Pliegue 1)")

# ------------------------------------------------------------------
# Construir ventanas de TRAIN y VAL del Pliegue 1
# ------------------------------------------------------------------
fold1 = particion["pliegues"][0]
train_start, train_end = fold1["train_idx"]
val_start, val_end = fold1["val_idx"]

X_train, meta_train = construir_ventanas(features, train_start, train_end, window=12)
X_val, meta_val = construir_ventanas(features, val_start, val_end, window=12)

print(f"\nVentanas de TRAIN (Pliegue 1): {X_train.shape}")
print(f"Ventanas de VAL   (Pliegue 1): {X_val.shape}")

# ------------------------------------------------------------------
# Verificaciones de correctitud
# ------------------------------------------------------------------
# 1) Conteo esperado: train pierde 11 pasos al inicio (no hay historia previa)
esperado_train = (train_end - max(train_start, 11)) * n_sensores
esperado_val = (val_end - val_start) * n_sensores
assert X_train.shape[0] == esperado_train, "Conteo de ventanas de train no coincide con lo esperado"
assert X_val.shape[0] == esperado_val, "Conteo de ventanas de val no coincide con lo esperado"
print("\nOK - el conteo de ventanas coincide exactamente con la formula esperada.")

# 2) Sin NaN ni Inf en ninguna ventana
assert not np.isnan(X_train).any() and not np.isnan(X_val).any(), "Hay NaN en las ventanas"
assert not np.isinf(X_train).any() and not np.isinf(X_val).any(), "Hay Inf en las ventanas"
print("OK - sin valores NaN o Inf en ninguna ventana.")

# 3) Verificacion manual: la primera ventana de train (sensor 0) debe
#    coincidir exactamente con features[0:12, 0, :]
manual = features[0:12, 0, :]
assert np.allclose(X_train[0], manual), "La primera ventana no coincide con el corte manual esperado"
print("OK - la primera ventana coincide exactamente con el corte manual de verificacion.")

# 4) Ninguna ventana de val debe tener t (instante final) fuera de [val_start, val_end)
assert meta_val[:, 1].min() == val_start and meta_val[:, 1].max() == val_end - 1, \
    "Las ventanas de val tienen instantes t fuera del rango esperado"
print("OK - todas las ventanas de val tienen su instante final dentro del rango correcto.")

print("\nSemana 3 - Tarea 3 completa. La funcion de ventaneo queda verificada y lista para "
      "reutilizarse en el entrenamiento de G0-G7 (Semana 7 en adelante).")
