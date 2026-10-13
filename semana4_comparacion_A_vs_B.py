"""
Semana 4 - Comparacion entre Esquemas A y B (PE1).

Responde a la Pregunta Especifica 1: como difieren la prevalencia, distribucion
y solapamiento de las etiquetas generadas por ambos esquemas.

Requiere: data/processed/labels_esquema_A.npz
          data/processed/labels_esquema_B.npz
          configs/particion_temporal.json
Genera:   results/eda/comparacion_A_vs_B.json
          results/eda/comparacion_A_vs_B.md
"""
import json
import numpy as np
from pathlib import Path

LABELS_A = Path("data/processed/labels_esquema_A.npz")
LABELS_B = Path("data/processed/labels_esquema_B.npz")
PARTICION = Path("configs/particion_temporal.json")
OUT_JSON = Path("results/eda/comparacion_A_vs_B.json")
OUT_MD = Path("results/eda/comparacion_A_vs_B.md")


def jaccard(a, b):
    inter = np.logical_and(a == 1, b == 1).sum()
    union = np.logical_or(a == 1, b == 1).sum()
    return float(inter / union) if union > 0 else 0.0


def estadisticas_bloque(bloque_labels, idx_a, idx_b, nombre):
    labels = bloque_labels[idx_a:idx_b, :]
    validos = labels != -1
    if validos.sum() == 0:
        return {"nombre": nombre, "n_instantes_validos": 0}
    n_anom = int((labels[validos] == 1).sum())
    total = int(validos.sum())
    return {
        "nombre": nombre,
        "rango_idx": [int(idx_a), int(idx_b)],
        "n_instantes_validos": total,
        "n_anomalos": n_anom,
        "prevalencia": float(n_anom / total),
    }


def main():
    print("Cargando etiquetas...")
    labels_A = np.load(LABELS_A)["labels"]
    labels_B = np.load(LABELS_B)["labels"]
    assert labels_A.shape == labels_B.shape, "Las formas de A y B no coinciden"
    print(f"  Forma: {labels_A.shape}")

    with open(PARTICION, "r", encoding="utf-8") as f:
        particion = json.load(f)

    reporte = {
        "descripcion": "Comparacion de los esquemas A y B para PE1",
        "prevalencia": {},
        "solapamiento": {},
        "distribucion_temporal_anomalias_por_sensor": {},
    }

    particiones = [("test", *particion["test_idx"])]
    for pl in particion["pliegues"]:
        k = pl["pliegue"]
        particiones.append((f"pliegue{k}_train", *pl["train_idx"]))
        particiones.append((f"pliegue{k}_val", *pl["val_idx"]))

    for nombre, ini, fin in particiones:
        st_a = estadisticas_bloque(labels_A, ini, fin, nombre)
        st_b = estadisticas_bloque(labels_B, ini, fin, nombre)
        reporte["prevalencia"][nombre] = {"A": st_a, "B": st_b}

    for nombre, ini, fin in particiones:
        a = labels_A[ini:fin, :]
        b = labels_B[ini:fin, :]
        mask = (a != -1) & (b != -1)
        if mask.sum() == 0:
            continue
        reporte["solapamiento"][nombre] = {
            "jaccard": jaccard(a[mask], b[mask]),
            "n_instantes_comparables": int(mask.sum()),
            "n_anomalos_A": int((a[mask] == 1).sum()),
            "n_anomalos_B": int((b[mask] == 1).sum()),
            "n_en_ambos": int(((a[mask] == 1) & (b[mask] == 1)).sum()),
        }

    horas = ((np.arange(labels_A.shape[0]) * 5) // 60) % 24
    dist = {"A_por_hora": [0] * 24, "B_por_hora": [0] * 24}
    for h in range(24):
        mask_h = horas == h
        dist["A_por_hora"][h] = int((labels_A[mask_h, :] == 1).sum())
        dist["B_por_hora"][h] = int((labels_B[mask_h, :] == 1).sum())
    reporte["distribucion_temporal_anomalias_por_sensor"] = dist

    def bloques_con_ambas_clases(labels):
        count = 0
        for nombre, ini, fin in particiones:
            bloque = labels[ini:fin, :]
            for s in range(bloque.shape[1]):
                serie = bloque[:, s]
                validos = serie != -1
                if validos.sum() == 0:
                    continue
                vals = serie[validos]
                if 0 in vals and 1 in vals:
                    count += 1
        return count

    reporte["bloques_con_ambas_clases"] = {
        "A": bloques_con_ambas_clases(labels_A),
        "B": bloques_con_ambas_clases(labels_B),
        "total_bloques_evaluables": sum((fin - ini) > 0 for _, ini, fin in particiones) * labels_A.shape[1],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    lines = ["# Comparacion Esquemas A vs B (PE1)\n"]
    lines.append("## Prevalencia por particion\n")
    lines.append("| Particion | A (%) | B (%) | Razon B/A |")
    lines.append("|---|---|---|---|")
    for nombre, st in reporte["prevalencia"].items():
        pa = st["A"].get("prevalencia", 0)
        pb = st["B"].get("prevalencia", 0)
        razon = pb / pa if pa > 0 else float("inf")
        lines.append(f"| {nombre} | {pa*100:.4f} | {pb*100:.4f} | {razon:.2f} |")

    lines.append("\n## Solapamiento (Jaccard)\n")
    lines.append("| Particion | Jaccard | A anomalos | B anomalos | En ambos |")
    lines.append("|---|---|---|---|---|")
    for nombre, st in reporte["solapamiento"].items():
        lines.append(f"| {nombre} | {st['jaccard']:.4f} | {st['n_anomalos_A']} | "
                     f"{st['n_anomalos_B']} | {st['n_en_ambos']} |")

    lines.append("\n## Bloques con ambas clases\n")
    b = reporte["bloques_con_ambas_clases"]
    lines.append(f"- Esquema A: {b['A']} bloques")
    lines.append(f"- Esquema B: {b['B']} bloques")
    lines.append(f"- Total evaluables: {b['total_bloques_evaluables']}")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReporte JSON: {OUT_JSON}")
    print(f"Reporte MD:   {OUT_MD}")
    print("\n--- Prevalencia resumida ---")
    for nombre, st in reporte["prevalencia"].items():
        print(f"  {nombre:22s}  A: {st['A'].get('prevalencia',0)*100:6.3f}%   "
              f"B: {st['B'].get('prevalencia',0)*100:6.3f}%")
    print("\nSemana 4 - Comparacion A vs B completa.")


if __name__ == "__main__":
    main()