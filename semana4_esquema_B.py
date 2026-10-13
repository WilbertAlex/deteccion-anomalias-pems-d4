"""
Semana 4 - Esquema B: inyeccion controlada de anomalias.

Implementa la Decision 9:
  - Para cada split (train, val, prueba) de cada pliegue, se inyectan
    anomalias con prevalencia 3% por bloque sensor x pliegue.
  - El perfil contextual (hora x dia) se estima SOLO con el train del pliegue.
  - La semilla de inyeccion es fija (42) para todas las arquitecturas.

Requiere: data/processed/pems04_semana2.npz
          configs/particion_temporal.json
          configs/config_esquema_B.json
          src/inyeccion_anomalias.py
Genera:   data/processed/labels_esquema_B.npz
          data/processed/features_esquema_B.npz
          results/eda/prevalencia_esquema_B.json
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("src")))
from inyeccion_anomalias import construir_perfil_hora_dia, inyectar_anomalias

DATA_PATH = Path("data/processed/pems04_semana2.npz")
PARTICION_PATH = Path("configs/particion_temporal.json")
CONFIG_B_PATH = Path("configs/config_esquema_B.json")
OUT_LABELS = Path("data/processed/labels_esquema_B.npz")
OUT_FEATURES = Path("data/processed/features_esquema_B.npz")
OUT_REPORT = Path("results/eda/prevalencia_esquema_B.json")


def cargar_config():
    with open(CONFIG_B_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def etiquetar_split(raw_split, idx_ini, perfil, cfg):
    raw_alt, labels = inyectar_anomalias(
        raw_split=raw_split,
        idx_inicio_split=idx_ini,
        perfil=perfil,
        prevalencia_objetivo=cfg["prevalencia_objetivo"],
        semilla=cfg["semilla_inyeccion"],
        distribucion_tipos=cfg["distribucion_tipos"],
        duraciones_colectivas=cfg["tipos_anomalia"]["colectiva"]["duraciones_pasos"],
        magnitud_sigma=cfg["tipos_anomalia"]["puntual"]["magnitud_sigma"],
        factor_reduccion=cfg["tipos_anomalia"]["colectiva"]["factor_reduccion"],
    )
    return raw_alt, labels


def main():
    cfg = cargar_config()
    print("Cargando dataset...")
    d = np.load(DATA_PATH)
    raw = d["data"]
    T, S, _ = raw.shape

    with open(PARTICION_PATH, "r", encoding="utf-8") as f:
        particion = json.load(f)

    raw_B = raw.copy()
    labels_B = np.full((T, S), -1, dtype=np.int8)

    reporte = {
        "regla": "inyeccion controlada 3% por bloque sensor x pliegue, semilla fija",
        "semilla_inyeccion": cfg["semilla_inyeccion"],
        "particiones": {},
    }

    print("\nProcesando pliegues...")
    for pl in particion["pliegues"]:
        k = pl["pliegue"]
        train_ini, train_fin = pl["train_idx"]
        val_ini, val_fin = pl["val_idx"]

        flow_train = raw[train_ini:train_fin, :, 0]
        perfil, _, _ = construir_perfil_hora_dia(flow_train)
        print(f"  Pliegue {k}: perfil construido con {len(perfil)} celdas")

        raw_train_alt, lab_train = etiquetar_split(
            raw[train_ini:train_fin], train_ini, perfil, cfg
        )
        raw_B[train_ini:train_fin] = raw_train_alt
        labels_B[train_ini:train_fin] = lab_train

        raw_val_alt, lab_val = etiquetar_split(
            raw[val_ini:val_fin], val_ini, perfil, cfg
        )
        raw_B[val_ini:val_fin] = raw_val_alt
        labels_B[val_ini:val_fin] = lab_val

        reporte["particiones"][f"pliegue{k}_train"] = {
            "rango_idx": [int(train_ini), int(train_fin)],
            "prevalencia": float(lab_train.mean()),
            "n_anomalos": int(lab_train.sum()),
            "n_instantes": int(lab_train.size),
        }
        reporte["particiones"][f"pliegue{k}_val"] = {
            "rango_idx": [int(val_ini), int(val_fin)],
            "prevalencia": float(lab_val.mean()),
            "n_anomalos": int(lab_val.sum()),
            "n_instantes": int(lab_val.size),
        }

    test_ini, test_fin = particion["test_idx"]
    flow_80 = raw[:test_ini, :, 0]
    perfil_test, _, _ = construir_perfil_hora_dia(flow_80)
    raw_test_alt, lab_test = etiquetar_split(
        raw[test_ini:test_fin], test_ini, perfil_test, cfg
    )
    raw_B[test_ini:test_fin] = raw_test_alt
    labels_B[test_ini:test_fin] = lab_test

    reporte["particiones"]["test"] = {
        "rango_idx": [int(test_ini), int(test_fin)],
        "prevalencia": float(lab_test.mean()),
        "n_anomalos": int(lab_test.sum()),
        "n_instantes": int(lab_test.size),
    }

    reporte["prevalencia_global"] = {
        "n_instantes": int(labels_B.size),
        "n_anomalos": int((labels_B == 1).sum()),
        "prevalencia": float((labels_B == 1).mean()),
    }

    print("\n--- Prevalencia por particion ---")
    for nombre, st in reporte["particiones"].items():
        print(f"  {nombre:22s}: {st['prevalencia']*100:6.3f}%  "
              f"({st['n_anomalos']:,}/{st['n_instantes']:,})")

    print(f"\n--- Prevalencia global ---")
    g = reporte["prevalencia_global"]
    print(f"  {g['prevalencia']*100:.3f}%  ({g['n_anomalos']:,}/{g['n_instantes']:,})")

    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_LABELS, labels=labels_B)
    print(f"\nLabels Esquema B guardados en {OUT_LABELS}")

    np.savez_compressed(OUT_FEATURES, data=raw_B)
    print(f"Features alteradas guardadas en {OUT_FEATURES}")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en {OUT_REPORT}")

    print("\nSemana 4 - Esquema B completo.")


if __name__ == "__main__":
    main()