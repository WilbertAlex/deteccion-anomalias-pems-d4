"""
Semana 5 - G0: linea base no parametrica, 12 configuraciones (4 ventanas x 3 multiplicadores).

REGLA DE SELECCION (fijada ANTES de ver resultados, Decision 10):
  PR-AUC se calcula sobre el score continuo |z| y por tanto NO depende del
  multiplicador k. Por eso la seleccion es en dos pasos, solo con VALIDACION:
    1) ventana w*  = argmax_w  media sobre los 3 pliegues de PR-AUC de validacion
    2) multiplicador k* = argmax_k media sobre los 3 pliegues de F1 de validacion (con w*)
  Luego UNA sola pasada sobre la prueba con (w*, k*).

Fuentes de datos por esquema:
  A: flujo crudo + labels_esquema_A.npz (identico en todos los pliegues).
  B: esquema_B_pliegue{k}.npz para validacion del pliegue k;
     esquema_B_final.npz para la prueba (generados por semana5_esquema_B_por_pliegue.py).

Requiere: data/processed/pems04_semana2.npz, labels_esquema_A.npz,
          esquema_B_pliegue{1,2,3}.npz, esquema_B_final.npz, configs/particion_temporal.json
Genera:   results/eda/g0_resultados.json y .md
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("src")))
from g0_linea_base import scores_g0, evaluar_split, evaluar_test_agregado

VENTANAS = [6, 11, 22, 44]
MULTIPLICADORES = [2.0, 2.5, 3.0]
P = Path("data/processed")


def main():
    part = json.loads(Path("configs/particion_temporal.json").read_text(encoding="utf-8"))
    raw_flow = np.load(P / "pems04_semana2.npz")["data"][:, :, 0]
    lab_A = np.load(P / "labels_esquema_A.npz")["labels"]
    te = tuple(part["test_idx"])

    # fuentes: esquema -> {pliegue_k: (flujo, labels), "final": (flujo, labels)}
    fuentes = {"A": {k: (raw_flow, lab_A) for k in (1, 2, 3, "final")}, "B": {}}
    for k in (1, 2, 3):
        z = np.load(P / f"esquema_B_pliegue{k}.npz")
        fuentes["B"][k] = (z["data"][:, :, 0], z["labels"])
    z = np.load(P / "esquema_B_final.npz")
    fuentes["B"]["final"] = (z["data"][:, :, 0], z["labels"])

    res = {"grilla": {"ventanas": VENTANAS, "multiplicadores": MULTIPLICADORES}, "esquemas": {}}
    for esq in ("A", "B"):
        print(f"\n=== Esquema {esq} ===")
        val = {w: [] for w in VENTANAS}
        for pl in part["pliegues"]:
            k = pl["pliegue"]
            flow, lab = fuentes[esq][k]
            a, b = pl["val_idx"]
            for w in VENTANAS:
                val[w].append(evaluar_split(scores_g0(flow, w), lab, a, b, MULTIPLICADORES))

        tabla = []
        for w in VENTANAS:
            tabla.append({"ventana": w,
                          "pr_auc_val_medio": float(np.nanmean([r["pr_auc"] for r in val[w]])),
                          "lift_val_medio": float(np.nanmean([r["lift"] for r in val[w]])),
                          "n_bloques_val": [r["n_bloques"] for r in val[w]],
                          "f1_val_medio": {str(kk): float(np.nanmean([r["f1"][kk] for r in val[w]]))
                                           for kk in MULTIPLICADORES}})
        w_star = max(tabla, key=lambda r: r["pr_auc_val_medio"])["ventana"]
        f1w = next(r for r in tabla if r["ventana"] == w_star)["f1_val_medio"]
        k_star = float(max(f1w, key=lambda kk: f1w[kk]))
        for r in tabla:
            print(f"  w={r['ventana']:>2}: PR-AUC val={r['pr_auc_val_medio']:.4f}  lift={r['lift_val_medio']:.1f}x  "
                  f"bloques={r['n_bloques_val']}  F1(k)=" + ", ".join(f"{kk}:{v:.3f}" for kk, v in r["f1_val_medio"].items()))
        print(f"  -> seleccion (solo validacion): w*={w_star}, k*={k_star}")

        flow_t, lab_t = fuentes[esq]["final"]
        sc = scores_g0(flow_t, w_star)
        ev = evaluar_split(sc, lab_t, te[0], te[1], [k_star])
        ag = evaluar_test_agregado(sc, lab_t, te[0], te[1], k_star)
        print(f"  TEST: PR-AUC={ev['pr_auc']:.4f} (prevalencia={ev['prevalencia']:.4f}, lift={ev['lift']:.1f}x) "
              f"F1={ev['f1'][k_star]:.3f} AUC-ROC={ag['auc_roc_agregado']:.3f} FPR={ag['fpr']:.4f} bloques={ev['n_bloques']}")
        res["esquemas"][esq] = {"validacion_por_ventana": tabla, "seleccion": {"ventana": w_star, "multiplicador": k_star},
                                "test": {"pr_auc": ev["pr_auc"], "prevalencia_media_bloque": ev["prevalencia"],
                                         "lift": ev["lift"], "f1": ev["f1"][k_star], "n_bloques": ev["n_bloques"], **ag}}

    Path("results/eda").mkdir(parents=True, exist_ok=True)
    Path("results/eda/g0_resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# G0 - resultados (seleccion solo en validacion; prueba evaluada una vez)\n",
         "PR-AUC = media sobre bloques (sensores con ambas clases). Lift = PR-AUC / prevalencia del bloque "
         "(azar = 1). **A: G0 comparte familia con el etiquetador; su resultado es techo del etiquetador, no evidencia de deteccion.**\n"]
    for esq in ("A", "B"):
        r = res["esquemas"][esq]; t = r["test"]
        L += [f"\n## Esquema {esq}\n", "| Ventana | PR-AUC val | Lift val | Bloques val (p1,p2,p3) | F1 k=2.0 | F1 k=2.5 | F1 k=3.0 |", "|---|---|---|---|---|---|---|"]
        for x in r["validacion_por_ventana"]:
            L.append(f"| {x['ventana']} | {x['pr_auc_val_medio']:.4f} | {x['lift_val_medio']:.1f} | {x['n_bloques_val']} | "
                     + " | ".join(f"{x['f1_val_medio'][str(kk)]:.3f}" for kk in MULTIPLICADORES) + " |")
        L += [f"\nSeleccion: w*={r['seleccion']['ventana']}, k*={r['seleccion']['multiplicador']}",
              f"\nPrueba: PR-AUC **{t['pr_auc']:.4f}** (prevalencia {t['prevalencia_media_bloque']:.4f}, lift {t['lift']:.1f}x), "
              f"F1 {t['f1']:.3f}, AUC-ROC {t['auc_roc_agregado']:.3f}, FPR {t['fpr']:.4f}, bloques evaluables {t['n_bloques']}"]
    Path("results/eda/g0_resultados.md").write_text("\n".join(L), encoding="utf-8")
    print("\nGuardado: results/eda/g0_resultados.json y .md\nSemana 5 - G0 completo.")


if __name__ == "__main__":
    main()
