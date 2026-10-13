"""
Semana 4 - Esquema A: etiquetado proxy causal por desviacion estadistica.

Implementa la Decision 8:
  - z-score causal: para cada instante t, mu y sigma se calculan sobre los
    288 pasos inmediatamente anteriores (t-288 a t-1) de la serie de FLUJO
    SIN NORMALIZAR (raw).
  - Umbral: abs(z) > 3.
  - Primeros 288 pasos de cada sensor: sin etiqueta (-1), excluidos de evaluacion.

Requiere: data/processed/pems04_semana2.npz
          configs/particion_temporal.json
          configs/config_esquema_A.json
Genera:   data/processed/labels_esquema_A.npz
          results/eda/prevalencia_esquema_A.json
"""
import json
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/pems04_semana2.npz")
PARTICION_PATH = Path("configs/particion_temporal.json")
CONFIG_A_PATH = Path("configs/config_esquema_A.json")
OUT_LABELS = Path("data/processed/labels_esquema_A.npz")
OUT_REPORT = Path("results/eda/prevalencia_esquema_A.json")


def cargar_config():
    with open(CONFIG_A_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def calcular_z_causal(flow, ventana):
    """z-score causal vectorizado con cumulative sums."""
    T, S = flow.shape
    cumsum = np.zeros((T + 1, S))
    cumsum2 = np.zeros((T + 1, S))
    cumsum[1:] = np.cumsum(flow, axis=0)
    cumsum2[1:] = np.cumsum(flow ** 2, axis=0)

    t_idx = np.arange(ventana, T)
    sum_w = cumsum[t_idx] - cumsum[t_idx - ventana]
    sum2_w = cumsum2[t_idx] - cumsum2[t_idx - ventana]

    mu = sum_w / ventana
    var = np.maximum(sum2_w / ventana - mu ** 2, 0.0)
    sigma = np.sqrt(var)

    flow_t = flow[t_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, (flow_t - mu) / sigma, 0.0)
    return z, t_idx


def reportar_prevalencia(labels, particion):
    reporte = {
        "regla": "abs(z) > 3 sobre ventana movil causal de 288 pasos de flujo raw",
        "primeros_pasos_sin_etiqueta": 288,
        "particiones": {},
    }

    def stats(idx_a, idx_b, nombre):
        bloque = labels[idx_a:idx_b, :]
        validos = bloque != -1
        if validos.sum() == 0:
            return {"nombre": nombre, "n_instantes": 0}
        n_anom = int((bloque[validos] == 1).sum())
        n_norm = int((bloque[validos] == 0).sum())
        total = n_anom + n_norm
        return {
            "nombre": nombre,
            "rango_idx": [int(idx_a), int(idx_b)],
            "n_instantes_validos": total,
            "n_anomalos": n_anom,
            "n_normales": n_norm,
            "prevalencia": float(n_anom / total) if total else 0.0,
        }

    reporte["particiones"]["test"] = stats(
        particion["test_idx"][0], particion["test_idx"][1], "test"
    )
    for pl in particion["pliegues"]:
        k = pl["pliegue"]
        reporte["particiones"][f"pliegue{k}_train"] = stats(
            pl["train_idx"][0], pl["train_idx"][1], f"pliegue{k}_train"
        )
        reporte["particiones"][f"pliegue{k}_val"] = stats(
            pl["val_idx"][0], pl["val_idx"][1], f"pliegue{k}_val"
        )

    prev_sensor = []
    for s in range(labels.shape[1]):
        serie = labels[:, s]
        validos = serie != -1
        if validos.sum() == 0:
            prev_sensor.append(None)
            continue
        n_anom = int((serie[validos] == 1).sum())
        total = int(validos.sum())
        prev_sensor.append({
            "sensor_idx": s,
            "n_instantes_validos": total,
            "n_anomalos": n_anom,
            "prevalencia": float(n_anom / total),
        })
    reporte["prevalencia_por_sensor"] = prev_sensor

    validos_global = labels != -1
    total_global = int(validos_global.sum())
    n_anom_global = int((labels[validos_global] == 1).sum())
    reporte["prevalencia_global"] = {
        "n_instantes_validos": total_global,
        "n_anomalos": n_anom_global,
        "prevalencia": float(n_anom_global / total_global) if total_global else 0.0,
    }
    return reporte


def main():
    cfg = cargar_config()
    ventana = cfg["ventana_movil_pasos"]
    umbral = cfg["umbral_z"]

    print(f"Cargando {DATA_PATH}...")
    d = np.load(DATA_PATH)
    raw = d["data"]
    flow = raw[:, :, 0]
    T, S = flow.shape
    print(f"  Forma del flujo: {flow.shape}")

    with open(PARTICION_PATH, "r", encoding="utf-8") as f:
        particion = json.load(f)

    print(f"\nCalculando z-scores causales (ventana={ventana}, umbral=abs(z)>{umbral})...")
    z, t_idx = calcular_z_causal(flow, ventana)
    print(f"  z-scores calculados para {len(t_idx)} instantes x {S} sensores")

    labels = np.full((T, S), -1, dtype=np.int8)
    labels[t_idx] = (np.abs(z) > umbral).astype(np.int8)

    assert np.all(labels[:ventana, :] == -1), "Los primeros 288 pasos deben estar sin etiqueta"
    assert np.all((labels[ventana:, :] == 0) | (labels[ventana:, :] == 1)), \
        "Despues de los 288 pasos solo debe haber 0 o 1"
    print("  OK: primeros 288 pasos sin etiqueta, resto etiquetado 0/1.")

    reporte = reportar_prevalencia(labels, particion)

    print("\n--- Prevalencia global ---")
    g = reporte["prevalencia_global"]
    print(f"  Instantes validos: {g['n_instantes_validos']:,}")
    print(f"  Instantes anomalos: {g['n_anomalos']:,}")
    print(f"  Prevalencia: {g['prevalencia']*100:.4f}%")

    print("\n--- Prevalencia por particion ---")
    for nombre, st in reporte["particiones"].items():
        if "prevalencia" in st:
            print(f"  {nombre:22s}: {st['prevalencia']*100:7.4f}%  "
                  f"({st['n_anomalos']:,}/{st['n_instantes_validos']:,})")

    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_LABELS, labels=labels)
    print(f"\nLabels guardados en {OUT_LABELS}")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en {OUT_REPORT}")
    print("\nSemana 4 - Esquema A completo.")


if __name__ == "__main__":
    main()