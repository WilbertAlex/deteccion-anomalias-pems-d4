"""
Semana 2 - Preprocesamiento: exclusion de sensores, variables temporales derivadas
y regla de hora pico. Aplica exactamente las decisiones registradas en
bitacora_decisiones_metodologicas.md y config_preprocesamiento.json.

Requiere: data/raw/pems04.npz (ya descargado y verificado en la Semana 1)
Genera:   data/processed/pems04_semana2.npz
          results/eda/sensores_excluidos.csv
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/pems04.npz")
CONFIG_PATH = Path("config_preprocesamiento.json")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("results/eda")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

raw = np.load(DATA_PATH)["data"]
n_steps, n_sensors, n_features = raw.shape
timestamps = pd.date_range(start="2018-01-01 00:00", periods=n_steps, freq="5min")
print(f"Cargado: {raw.shape}")

# ------------------------------------------------------------------
# 1) Exclusion de sensores (Decision 3)
# ------------------------------------------------------------------
flow = raw[:, :, 0]
cond = config["criterio_exclusion_sensores"]["condiciones"]
umbral_ceros = cond[0]["umbral_pct"]
umbral_racha = cond[1]["umbral_pasos"]

zero_pct = (flow == 0).mean(axis=0) * 100
excluidos_ceros = set(np.where(zero_pct > umbral_ceros)[0])

excluidos_racha = set()
for s in range(n_sensors):
    f = flow[:, s]
    changes = np.diff(f) != 0
    idx = np.where(np.concatenate(([True], changes, [True])))[0]
    if np.diff(idx).max() >= umbral_racha:
        excluidos_racha.add(s)

excluidos = sorted(excluidos_ceros | excluidos_racha)
retenidos = [s for s in range(n_sensors) if s not in excluidos]

print(f"Sensores excluidos: {len(excluidos)} | Retenidos: {len(retenidos)}")
assert len(excluidos) == config["criterio_exclusion_sensores"]["resultado"]["sensores_excluidos"], \
    "El numero de sensores excluidos no coincide con lo documentado en la bitacora - revisar"

# Guardar listado de exclusion con la razon
registro = []
for s in range(n_sensors):
    razon = []
    if s in excluidos_ceros:
        razon.append(f">{umbral_ceros}% ceros")
    if s in excluidos_racha:
        razon.append(f"racha congelada >= {umbral_racha} pasos")
    registro.append({
        "sensor_id": s,
        "excluido": s in excluidos,
        "razon": "; ".join(razon) if razon else "",
    })
pd.DataFrame(registro).to_csv(RESULTS_DIR / "sensores_excluidos.csv", index=False)

# Arreglo filtrado: solo sensores retenidos
raw_filtrado = raw[:, retenidos, :]
print(f"Forma tras filtrar: {raw_filtrado.shape}")

# ------------------------------------------------------------------
# 2) Variables temporales derivadas (codificacion ciclica)
# ------------------------------------------------------------------
hour = timestamps.hour.values
dow = timestamps.dayofweek.values  # 0=lunes ... 6=domingo

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
day_sin = np.sin(2 * np.pi * dow / 7)
day_cos = np.cos(2 * np.pi * dow / 7)

# ------------------------------------------------------------------
# 3) Indicador de hora pico (Decision 4)
# ------------------------------------------------------------------
horas_pico = set(config["regla_hora_pico"]["horas_pico"])
is_peak_hour = np.array([1 if h in horas_pico else 0 for h in hour])

print(f"\nVariables temporales construidas para {n_steps} pasos:")
print(f"  hour_sin, hour_cos, day_sin, day_cos, is_peak_hour")
print(f"  Pasos marcados como hora pico: {is_peak_hour.sum()} de {n_steps} "
      f"({is_peak_hour.mean()*100:.1f}%)")

# ------------------------------------------------------------------
# 4) Guardar todo junto
# ------------------------------------------------------------------
np.savez_compressed(
    OUT_DIR / "pems04_semana2.npz",
    data=raw_filtrado,              # (16992, 291, 3)
    sensores_retenidos=np.array(retenidos),
    hour_sin=hour_sin,
    hour_cos=hour_cos,
    day_sin=day_sin,
    day_cos=day_cos,
    is_peak_hour=is_peak_hour,
)
print(f"\nGuardado en {OUT_DIR / 'pems04_semana2.npz'}")
print(f"Listado de exclusion guardado en {RESULTS_DIR / 'sensores_excluidos.csv'}")
print("\nSemana 2 (Tareas 1-3) completa.")
