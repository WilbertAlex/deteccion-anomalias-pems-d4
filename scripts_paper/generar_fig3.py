"""
Genera la Figura 3 del short paper: serie de flujo vehicular de un sensor
con las anomalias marcadas por ambos esquemas.

Se selecciona automaticamente el sensor con mayor numero de anomalias
del Esquema A en el rango visualizado, para asegurar que se vean puntos
de ambos esquemas.

Requiere: data/processed/pems04_semana2.npz
          data/processed/labels_esquema_A.npz
          data/processed/labels_esquema_B.npz
Genera:   docs/paper/fig3_serie_ejemplo.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})

DATA = Path("data/processed/pems04_semana2.npz")
LABELS_A = Path("data/processed/labels_esquema_A.npz")
LABELS_B = Path("data/processed/labels_esquema_B.npz")
OUTPUT = Path("docs/paper/fig3_serie_ejemplo.png")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Cargar datos
d = np.load(DATA)
flow = d["data"][:, :, 0]  # (16992, 291)
labels_A = np.load(LABELS_A)["labels"]
labels_B = np.load(LABELS_B)["labels"]

# Rango de visualizacion: 3 dias (864 pasos) empezando en el paso 5000
# (evita los primeros 288 sin etiqueta en el Esquema A)
T_INI, T_FIN = 5000, 5864
VENTANA = T_FIN - T_INI

# Buscar el sensor con mas anomalias A en la ventana
anomalias_A_ventana = (labels_A[T_INI:T_FIN, :] == 1).sum(axis=0)
# Filtrar: tambien debe tener anomalias B
anomalias_B_ventana = (labels_B[T_INI:T_FIN, :] == 1).sum(axis=0)
candidatos = np.where((anomalias_A_ventana > 0) & (anomalias_B_ventana > 0))[0]

if len(candidatos) == 0:
    print("ADVERTENCIA: ningun sensor tiene ambos tipos de anomalias en la ventana.")
    print("Usando el sensor con mas anomalias A.")
    sensor = int(np.argmax(anomalias_A_ventana))
else:
    # Elegir el de mas anomalias A entre los candidatos
    sensor = int(candidatos[np.argmax(anomalias_A_ventana[candidatos])])

n_A = int(anomalias_A_ventana[sensor])
n_B = int(anomalias_B_ventana[sensor])
print(f"Sensor seleccionado: {sensor}")
print(f"  Anomalias A en ventana: {n_A}")
print(f"  Anomalias B en ventana: {n_B}")

# Crear figura
fig, ax = plt.subplots(figsize=(7, 2.4))
t = np.arange(T_INI, T_FIN)

# Serie principal
ax.plot(
    t, flow[T_INI:T_FIN, sensor],
    color="#333333", linewidth=0.7,
    label="Flujo vehicular", zorder=1
)

# Marcas Esquema A (circulos rojos)
mask_A = labels_A[T_INI:T_FIN, sensor] == 1
ax.scatter(
    t[mask_A], flow[T_INI:T_FIN, sensor][mask_A],
    color="#E63946", s=28, marker="o", zorder=3,
    edgecolors="black", linewidths=0.4,
    label=f"Esquema A (|z|>3)"
)

# Marcas Esquema B (triangulos azules)
mask_B = labels_B[T_INI:T_FIN, sensor] == 1
ax.scatter(
    t[mask_B], flow[T_INI:T_FIN, sensor][mask_B],
    color="#2E86AB", s=28, marker="^", zorder=2,
    edgecolors="black", linewidths=0.4,
    label=f"Esquema B (inyección)"
)

ax.set_xlabel("Paso temporal (5 min)")
ax.set_ylabel("Flujo (veh/5 min)")
ax.grid(alpha=0.3, linestyle=":", linewidth=0.5)
ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8)

# Anotar el rango temporal en formato fecha
fecha_ini = "2018-01-18"  # paso 5000 = 5000*5min = 25000 min = ~17.4 dias
ax.set_xlim(T_INI, T_FIN)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigura 3 generada: {OUTPUT}")
print(f"Tamano: {OUTPUT.stat().st_size / 1024:.1f} KB")