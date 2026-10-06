"""
Genera la Figura 2 del short paper: distribucion horaria de anomalias
para los Esquemas A y B.

Requiere: results/eda/comparacion_A_vs_B.json
Genera:   docs/paper/fig2_distribucion_horaria.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuracion de estilo IEEE
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})

INPUT = Path("results/eda/comparacion_A_vs_B.json")
OUTPUT = Path("docs/paper/fig2_distribucion_horaria.png")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Cargar datos reales
with open(INPUT, "r", encoding="utf-8") as f:
    reporte = json.load(f)

dist = reporte["distribucion_temporal_anomalias_por_sensor"]
A_por_hora = np.array(dist["A_por_hora"])
B_por_hora = np.array(dist["B_por_hora"])
horas = np.arange(24)

# Validacion
assert len(A_por_hora) == 24, "A_por_hora debe tener 24 elementos"
assert len(B_por_hora) == 24, "B_por_hora debe tener 24 elementos"
print(f"Total anomalias A: {A_por_hora.sum():,}")
print(f"Total anomalias B: {B_por_hora.sum():,}")

# Crear figura
fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

# Barras del Esquema A (eje izquierdo)
barras = ax1.bar(
    horas - 0.2, A_por_hora, width=0.4,
    color="#2E86AB", edgecolor="black", linewidth=0.3,
    label="Esquema A (0.10%)"
)
ax1.set_xlabel("Hora del día")
ax1.set_ylabel("Anomalías — Esquema A", color="#2E86AB", fontsize=8)
ax1.tick_params(axis="y", labelcolor="#2E86AB")
ax1.set_xticks(range(0, 24, 3))
ax1.set_xlim(-0.6, 23.6)
ax1.grid(axis="y", alpha=0.3, linestyle=":")

# Linea del Esquema B (eje derecho)
ax2 = ax1.twinx()
linea, = ax2.plot(
    horas, B_por_hora, marker="o", markersize=3,
    linewidth=1.4, color="#E63946",
    label="Esquema B (3.03%)"
)
ax2.set_ylabel("Anomalías — Esquema B", color="#E63946", fontsize=8)
ax2.tick_params(axis="y", labelcolor="#E63946")

# Leyenda unificada
handles = [barras, linea]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc="upper left", frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigura 2 generada: {OUTPUT}")
print(f"Tamano: {OUTPUT.stat().st_size / 1024:.1f} KB")