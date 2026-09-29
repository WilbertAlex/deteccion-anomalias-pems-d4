"""
Análisis exploratorio inicial de PeMS04.
Tarea de la Semana 1 asignada a Prieto.

Requiere que data/raw/pems04.npz ya exista (ejecutar antes download_pems04.sh).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/raw/pems04.npz")
OUT_DIR = Path("results/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Cargar datos
# ------------------------------------------------------------------
raw = np.load(DATA_PATH)["data"]  # forma (16992, 307, 3)
n_steps, n_sensors, n_features = raw.shape
feature_names = ["flow", "occupancy", "speed"]

print(f"Forma del arreglo: {raw.shape}")
print(f"Pasos temporales: {n_steps} | Sensores: {n_sensors} | Variables: {n_features}")

# Construir índice temporal: 5 minutos por paso, desde el 1 de enero de 2018
timestamps = pd.date_range(start="2018-01-01 00:00", periods=n_steps, freq="5min")
print(f"Rango de fechas: {timestamps[0]} a {timestamps[-1]}")

# ------------------------------------------------------------------
# Estadística descriptiva por variable (agregando todos los sensores)
# ------------------------------------------------------------------
summary = {}
for i, name in enumerate(feature_names):
    col = raw[:, :, i].flatten()
    summary[name] = {
        "media": np.mean(col),
        "desv_std": np.std(col),
        "min": np.min(col),
        "max": np.max(col),
        "pct_ceros": float(np.mean(col == 0) * 100),
    }

summary_df = pd.DataFrame(summary).T
print("\nResumen estadístico global por variable:")
print(summary_df)
summary_df.to_csv(OUT_DIR / "resumen_estadistico_global.csv")

# ------------------------------------------------------------------
# Estadística por sensor (para las tres variables)
# ------------------------------------------------------------------
per_sensor = pd.DataFrame({
    "sensor_id": range(n_sensors),
    "flow_mean": raw[:, :, 0].mean(axis=0),
    "flow_std": raw[:, :, 0].std(axis=0),
    "occupancy_mean": raw[:, :, 1].mean(axis=0),
    "speed_mean": raw[:, :, 2].mean(axis=0),
})
per_sensor.to_csv(OUT_DIR / "resumen_por_sensor.csv", index=False)
print(f"\nResumen por sensor guardado en {OUT_DIR / 'resumen_por_sensor.csv'}")

# ------------------------------------------------------------------
# Valores faltantes (ceros pueden indicar sensor caído, no tráfico real)
# ------------------------------------------------------------------
zero_flow_pct = np.mean(raw[:, :, 0] == 0) * 100
print(f"\nPorcentaje de observaciones de flujo en cero: {zero_flow_pct:.2f}%")
print("(revisar si corresponde a sensores caídos o a tráfico nulo real de madrugada)")

# ------------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------------
# 1) Distribución de flujo (todos los sensores)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, name in enumerate(feature_names):
    axes[i].hist(raw[:, :, i].flatten(), bins=80, color="#2a78d6", alpha=0.85)
    axes[i].set_title(f"Distribución de {name}")
    axes[i].set_xlabel(name)
    axes[i].set_ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(OUT_DIR / "distribuciones_globales.png", dpi=150)
plt.close()

# 2) Serie temporal de un sensor de ejemplo (primera semana)
sensor_idx = 0
one_week = 12 * 24 * 7  # 12 pasos/hora x 24h x 7 días
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(timestamps[:one_week], raw[:one_week, sensor_idx, 0], color="#2a78d6", linewidth=0.9)
ax.set_title(f"Flujo vehicular — sensor {sensor_idx} (primera semana)")
ax.set_xlabel("Fecha")
ax.set_ylabel("Flujo")
plt.tight_layout()
plt.savefig(OUT_DIR / f"serie_temporal_sensor_{sensor_idx}.png", dpi=150)
plt.close()

print(f"\nGráficos guardados en {OUT_DIR}/")
print("EDA inicial completo.")
