"""
Semana 3 - Tarea 1: Particion temporal walk-forward.

Regla (Seccion 2.4 del protocolo):
- Ultimo 20% cronologico = conjunto de prueba final (NO se toca hasta evaluacion final)
- 80% restante se divide en 4 bloques iguales para construir 3 pliegues expansivos:
    Pliegue 1: train = bloque 1           | val = bloque 2
    Pliegue 2: train = bloques 1-2        | val = bloque 3
    Pliegue 3: train = bloques 1-2-3      | val = bloque 4

Requiere: data/processed/pems04_semana2.npz (generado en la Semana 2)
Genera:   configs/particion_temporal.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/pems04_semana2.npz")
OUT_PATH = Path("configs/particion_temporal.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

d = np.load(DATA_PATH)
raw = d["data"]  # (16992, 291, 3)
n_steps = raw.shape[0]
timestamps = pd.date_range(start="2018-01-01 00:00", periods=n_steps, freq="5min")

# ------------------------------------------------------------------
# 1) Conjunto de prueba final: ultimo 20%
# ------------------------------------------------------------------
test_start_idx = int(n_steps * 0.8)
pre_test_idx = test_start_idx  # los primeros pre_test_idx pasos son pre-test (80%)

print(f"Total de pasos: {n_steps}")
print(f"Conjunto de prueba final: indices [{test_start_idx}, {n_steps}) "
      f"-> {timestamps[test_start_idx]} a {timestamps[-1]}")
print(f"  ({n_steps - test_start_idx} pasos, {(n_steps - test_start_idx)/n_steps*100:.1f}%)")

# ------------------------------------------------------------------
# 2) Dividir el 80% restante en 4 bloques iguales
# ------------------------------------------------------------------
block_size = pre_test_idx // 4
bloques = []
for i in range(4):
    start = i * block_size
    end = (i + 1) * block_size if i < 3 else pre_test_idx  # el 4to bloque toma el resto
    bloques.append((start, end))
    print(f"Bloque {i+1}: indices [{start}, {end}) -> "
          f"{timestamps[start]} a {timestamps[end-1]} ({end-start} pasos)")

# ------------------------------------------------------------------
# 3) Construir los 3 pliegues expansivos
# ------------------------------------------------------------------
pliegues = []
for k in range(3):
    train_start, train_end = bloques[0][0], bloques[k][1]
    val_start, val_end = bloques[k + 1][0], bloques[k + 1][1]
    pliegue = {
        "pliegue": k + 1,
        "train_idx": [train_start, train_end],
        "train_fechas": [str(timestamps[train_start]), str(timestamps[train_end - 1])],
        "val_idx": [val_start, val_end],
        "val_fechas": [str(timestamps[val_start]), str(timestamps[val_end - 1])],
    }
    pliegues.append(pliegue)
    print(f"\nPliegue {k+1}:")
    print(f"  Train: [{train_start}, {train_end}) -> {timestamps[train_start]} a {timestamps[train_end-1]} "
          f"({train_end-train_start} pasos)")
    print(f"  Val:   [{val_start}, {val_end}) -> {timestamps[val_start]} a {timestamps[val_end-1]} "
          f"({val_end-val_start} pasos)")

# ------------------------------------------------------------------
# 4) Verificacion de no-solapamiento y no-fuga (control de sanidad)
# ------------------------------------------------------------------
print("\nVerificacion de integridad de la particion:")
for p in pliegues:
    assert p["train_idx"][1] <= p["val_idx"][0], \
        f"Pliegue {p['pliegue']}: el train se traslapa con val - FUGA DE DATOS"
    assert p["val_idx"][1] <= test_start_idx, \
        f"Pliegue {p['pliegue']}: val se traslapa con el conjunto de prueba - FUGA DE DATOS"
print("  OK - ningun pliegue usa informacion del futuro ni del conjunto de prueba final.")

# ------------------------------------------------------------------
# 5) Guardar configuracion
# ------------------------------------------------------------------
particion = {
    "n_steps_total": int(n_steps),
    "test_idx": [test_start_idx, n_steps],
    "test_fechas": [str(timestamps[test_start_idx]), str(timestamps[-1])],
    "bloques_pre_test": [{"bloque": i+1, "idx": list(b)} for i, b in enumerate(bloques)],
    "pliegues": pliegues,
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(particion, f, indent=2, ensure_ascii=False)

print(f"\nParticion guardada en {OUT_PATH}")
print("Semana 3 - Tarea 1 completa.")
