"""
Semana 5 - Comparacion A vs B (PE1) con el Esquema B CORREGIDO (por pliegue).

Reemplaza a semana4_comparacion_A_vs_B.py, que usaba el arreglo unico de B
(ver Decision 10). Unidades de comparacion = bloques de EVALUACION:
validacion de los pliegues 1-3 y prueba final (4 splits x 291 sensores = 1 164 bloques).
Solo cuentan pasos con etiqueta A valida (se excluyen los 288 primeros).

Genera: results/eda/comparacion_A_vs_B_v2.json y .md
"""
import json
import numpy as np
from pathlib import Path

P = Path("data/processed")
part = json.loads(Path("configs/particion_temporal.json").read_text(encoding="utf-8"))
A = np.load(P / "labels_esquema_A.npz")["labels"]

splits = [(f"val_pliegue{p['pliegue']}", p["val_idx"], f"esquema_B_pliegue{p['pliegue']}") for p in part["pliegues"]]
splits.append(("prueba", part["test_idx"], "esquema_B_final"))

rep = {"splits": {}, "agregado": {}}
horas_A = np.zeros(24, dtype=int); horas_B = np.zeros(24, dtype=int)
tot = bA = bB = 0
ev_A, ev_B = [], []
L = ["# Comparacion Esquemas A vs B (v2, B por pliegue)\n",
     "| Split | Pasos validos | Anom. A | Anom. B | A (%) | B (%) | En ambos | Jaccard | Bloques ambas clases A | Bloques ambas clases B |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for nombre, (a, b), fb in splits:
    B = np.load(P / f"{fb}.npz")["labels"]
    la, lb = A[a:b], B[a:b]
    ok = la != -1
    ia, ib = (la == 1) & ok, (lb == 1) & ok
    inter, uni = int((ia & ib).sum()), int((ia | ib).sum())
    ca = sum(1 for s in range(la.shape[1]) if len(np.unique(la[:, s][ok[:, s]])) == 2)
    cb = sum(1 for s in range(la.shape[1]) if len(np.unique(lb[:, s][ok[:, s]])) == 2)
    n = int(ok.sum())
    rep["splits"][nombre] = {"pasos_validos": n, "anomalos_A": int(ia.sum()), "anomalos_B": int(ib.sum()),
                             "prevalencia_A": float(ia.sum() / n), "prevalencia_B": float(ib.sum() / n),
                             "interseccion": inter, "jaccard": inter / uni if uni else 0.0,
                             "bloques_ambas_clases_A": ca, "bloques_ambas_clases_B": cb,
                             "bloques_totales": la.shape[1]}
    r = rep["splits"][nombre]
    L.append(f"| {nombre} | {n:,} | {r['anomalos_A']:,} | {r['anomalos_B']:,} | {r['prevalencia_A']*100:.3f} | "
             f"{r['prevalencia_B']*100:.3f} | {inter} | {r['jaccard']:.5f} | {ca} | {cb} |")
    tot += la.shape[1]; bA += ca; bB += cb
    idx = np.arange(a, b); h = ((idx * 5) // 60) % 24
    for hh in range(24):
        horas_A[hh] += int(ia[h == hh].sum()); horas_B[hh] += int(ib[h == hh].sum())

rep["agregado"] = {"bloques_totales": tot, "bloques_ambas_clases_A": bA, "bloques_ambas_clases_B": bB,
                   "fraccion_A": bA / tot, "fraccion_B": bB / tot,
                   "anomalias_por_hora_A": horas_A.tolist(), "anomalias_por_hora_B": horas_B.tolist()}
L += ["", f"**Bloques con ambas clases**: A {bA} de {tot} ({bA/tot:.1%}); B {bB} de {tot} ({bB/tot:.1%}).",
      "", "Implicacion: bajo A, la fraccion indicada de bloques es evaluable para PR-AUC; el contraste de "
      "Friedman por bloques sensor x pliegue tendra menos bloques bajo A que bajo B.", "",
      "## Anomalias por hora del dia (suma de los 4 splits)", "| Hora | A | B |", "|---|---|---|"]
L += [f"| {h:02d} | {horas_A[h]} | {horas_B[h]} |" for h in range(24)]
Path("results/eda").mkdir(parents=True, exist_ok=True)
Path("results/eda/comparacion_A_vs_B_v2.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
Path("results/eda/comparacion_A_vs_B_v2.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L[:12]))
print(f"\nBloques ambas clases: A {bA}/{tot} ({bA/tot:.1%}) | B {bB}/{tot}")
print("Guardado: results/eda/comparacion_A_vs_B_v2.json y .md")
