"""
Semana 6 - Ampliacion de la salida de G0 (hallazgos H-03, H-04, H-06, H-08 de la revision del 13 oct.).

NO cambia la seleccion de G0 (Decision 11): lee (w*, k*) de results/eda/g0_resultados.json.
Agrega, sin tocar nada anterior:
  1) Tabla larga por bloque (condicion, esquema, split, sensor, pr_auc, ...) con el formato que
     usaran G1-G7 (insumo de Friedman/Holm/Nemenyi). Incluye G0 con w* y G0 con w=11
     (analisis de sensibilidad: 11 pasos es lo maximo que cabe en la ventana de 12 de las redes).
  2) Umbral de recall 0.90: se fija en la validacion del pliegue 3 (PPI 2.7.1 paso 4) y se aplica
     a la prueba; se reporta recall y FPR. Ademas, k* por F1 solo en el pliegue 3, para comparar
     con el k* de la Decision 11 (media de 3 pliegues).
  3) Esquema B: composicion de los instantes anomalos por tipo y PR-AUC de G0 por tipo de anomalia.
  4) Normalizacion bajo el Esquema B: diferencia entre mu/sigma de la serie original y de la serie
     inyectada del train de cada pliegue (insumo de la Decision 14).
Controles (assert) al final de cada bloque.

Requiere: data/processed/{pems04_semana2,labels_esquema_A,esquema_B_pliegue1..3,esquema_B_final}.npz
          (los esquema_B_* con la clave 'tipos'), configs/particion_temporal.json,
          results/eda/g0_resultados.json
Genera:   results/eda/g0_por_bloque.csv, results/eda/g0_ampliado.json, results/eda/g0_ampliado.md
"""
import csv
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path("src")))
from g0_linea_base import scores_g0, evaluar_split

P = Path("data/processed")
R = Path("results/eda")
W_SENS = 11
NOMBRE_TIPO = {1: "puntual", 2: "contextual", 3: "colectiva"}


def pr_bloque(y, sc):
    ok = (y != -1) & ~np.isnan(sc)
    y, sc = y[ok], sc[ok]
    if len(y) == 0 or y.min() == y.max():
        return float("nan"), int(len(y)), int((y == 1).sum())
    return float(average_precision_score(y, sc)), int(len(y)), int((y == 1).sum())


def umbral_recall(y, sc, recall=0.90):
    """Mayor umbral tal que recall >= `recall` sobre los positivos validos (pred = score >= umbral)."""
    ok = (y != -1) & ~np.isnan(sc)
    pos = np.sort(sc[ok & (y == 1)])
    idx = int(np.floor((1 - recall) * len(pos)))
    return float(pos[idx])


def main():
    part = json.loads(Path("configs/particion_temporal.json").read_text(encoding="utf-8-sig"))
    g0 = json.loads((R / "g0_resultados.json").read_text(encoding="utf-8"))
    raw = np.load(P / "pems04_semana2.npz")["data"]
    flow = raw[:, :, 0]
    lab_A = np.load(P / "labels_esquema_A.npz")["labels"]
    te = tuple(part["test_idx"])
    S = flow.shape[1]

    fuentes = {"A": {k: (flow, lab_A, None) for k in (1, 2, 3, "final")}, "B": {}}
    for k in (1, 2, 3):
        z = np.load(P / f"esquema_B_pliegue{k}.npz")
        assert "tipos" in z.files, "Falta 'tipos': vuelve a correr semana5_esquema_B_por_pliegue.py (version nueva)"
        fuentes["B"][k] = (z["data"][:, :, 0], z["labels"], z["tipos"])
    z = np.load(P / "esquema_B_final.npz")
    fuentes["B"]["final"] = (z["data"][:, :, 0], z["labels"], z["tipos"])

    splits = [(f"val_pliegue{pl['pliegue']}", pl["pliegue"], tuple(pl["val_idx"])) for pl in part["pliegues"]]
    splits.append(("prueba", "final", te))

    filas, salida = [], {"nota": "G0 no se reselecciona: (w*, k*) vienen de g0_resultados.json (Decision 11)."}

    # ---------- 1) tabla larga por bloque ----------
    for esq in ("A", "B"):
        w_star = g0["esquemas"][esq]["seleccion"]["ventana"]
        for cond, w in ((f"G0_w{w_star}", w_star), (f"G0_w{W_SENS}_sensibilidad", W_SENS)):
            cache = {}
            for nombre, key, (a, b) in splits:
                fl, lab, _ = fuentes[esq][key]
                if key not in cache:
                    cache[key] = scores_g0(fl, w)
                sc = cache[key]
                for s in range(S):
                    pr, n, npos = pr_bloque(lab[a:b, s], sc[a:b, s])
                    filas.append({"condicion": cond, "esquema": esq, "split": nombre, "sensor_idx": s,
                                  "pr_auc": "" if np.isnan(pr) else f"{pr:.10f}",
                                  "n_validos": n, "n_positivos": npos, "evaluable": int(not np.isnan(pr))})
    R.mkdir(parents=True, exist_ok=True)
    with open(R / "g0_por_bloque.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0].keys())); wr.writeheader(); wr.writerows(filas)
    resumen = {}
    for esq in ("A", "B"):
        for sp in ("val", "prueba"):
            rs = [r for r in filas if r["esquema"] == esq and r["condicion"].startswith("G0_w") and not r["condicion"].endswith("sensibilidad")
                  and (r["split"].startswith("val") if sp == "val" else r["split"] == "prueba")]
            ev = [r for r in rs if r["evaluable"]]
            resumen[f"{esq}_{sp}"] = {"bloques_totales": len(rs), "bloques_evaluables": len(ev),
                                      "pr_auc_medio": float(np.mean([float(r["pr_auc"]) for r in ev]))}
    salida["tabla_larga"] = {"archivo": "results/eda/g0_por_bloque.csv", "filas": len(filas), "resumen": resumen}
    # control: la tabla larga reproduce g0_resultados.json
    for esq in ("A", "B"):
        t = g0["esquemas"][esq]
        assert abs(resumen[f"{esq}_prueba"]["pr_auc_medio"] - t["test"]["pr_auc"]) < 1e-8, f"{esq}: PR-AUC prueba no coincide"
        assert resumen[f"{esq}_prueba"]["bloques_evaluables"] == t["test"]["n_bloques"], f"{esq}: bloques de prueba no coinciden"
        vm = np.mean([x["pr_auc_val_medio"] for x in t["validacion_por_ventana"] if x["ventana"] == t["seleccion"]["ventana"]])
        pf = [np.mean([float(r["pr_auc"]) for r in filas if r["esquema"] == esq and r["split"] == f"val_pliegue{k}"
                       and r["condicion"] == f"G0_w{t['seleccion']['ventana']}" and r["evaluable"]]) for k in (1, 2, 3)]
        assert abs(np.mean(pf) - vm) < 1e-8, f"{esq}: PR-AUC val no coincide con g0_resultados.json"
    print("1) Tabla larga:", len(filas), "filas; coincide con g0_resultados.json (PR-AUC val y prueba, n bloques)")

    # ---------- 2) umbral de recall 0.90 y k* solo en el pliegue 3 ----------
    salida["recall090"] = {}
    for esq in ("A", "B"):
        w_star = g0["esquemas"][esq]["seleccion"]["ventana"]
        k_star = g0["esquemas"][esq]["seleccion"]["multiplicador"]
        out = {}
        for cond, w in (("w_star", w_star), ("w11_sensibilidad", W_SENS)):
            fl3, lab3, _ = fuentes[esq][3]; a3, b3 = tuple(part["pliegues"][2]["val_idx"])
            sc3 = scores_g0(fl3, w)
            thr = umbral_recall(lab3[a3:b3].ravel(), sc3[a3:b3].ravel())
            flt, labt, _ = fuentes[esq]["final"]
            sct = scores_g0(flt, w)
            y, s_ = labt[te[0]:te[1]].ravel(), sct[te[0]:te[1]].ravel()
            ok = (y != -1) & ~np.isnan(s_); y, s_ = y[ok], s_[ok]
            pred = s_ >= thr
            rec = float((pred & (y == 1)).sum() / (y == 1).sum())
            fpr = float((pred & (y == 0)).sum() / (y == 0).sum())
            y3 = lab3[a3:b3].ravel(); s3 = sc3[a3:b3].ravel(); ok3 = (y3 != -1) & ~np.isnan(s3)
            rec3 = float(((s3[ok3] >= thr) & (y3[ok3] == 1)).sum() / (y3[ok3] == 1).sum())
            assert rec3 >= 0.90 - 1e-9, "el umbral no alcanza recall 0.90 en la validacion del pliegue 3"
            ev3 = evaluar_split(sc3, lab3, a3, b3, [2.0, 2.5, 3.0])
            k_f3 = max(ev3["f1"], key=lambda kk: ev3["f1"][kk])
            out[cond] = {"ventana": w, "umbral_recall090_val_pliegue3": thr, "recall_val_pliegue3": rec3,
                         "recall_prueba": rec, "fpr_prueba_a_ese_umbral": fpr,
                         "k_por_F1_solo_pliegue3": float(k_f3), "k_decision11_media_3_pliegues": k_star if w == w_star else None}
            print(f"2) {esq} {cond:>16}: umbral={thr:.3f}  recall prueba={rec:.3f}  FPR prueba={fpr:.4f}  k(F1, pliegue 3)={k_f3}")
        salida["recall090"][esq] = out

    # ---------- 3) Esquema B: composicion y PR-AUC por tipo ----------
    comp, por_tipo = {}, {}
    w_star = g0["esquemas"]["B"]["seleccion"]["ventana"]
    for nombre, key, (a, b) in splits:
        fl, lab, tip = fuentes["B"][key]
        t = tip[a:b]; n = int((t > 0).sum())
        comp[nombre] = {NOMBRE_TIPO[c]: float((t == c).sum() / n) for c in (1, 2, 3)}
        assert abs(sum(comp[nombre].values()) - 1) < 1e-9
    salida["B_composicion_instantes"] = comp
    for cond, w in ((f"G0_w{w_star}", w_star), (f"G0_w{W_SENS}_sensibilidad", W_SENS)):
        por_tipo[cond] = {}
        for nombre, key, (a, b) in splits:
            if nombre not in ("prueba", "val_pliegue3"):
                continue
            fl, lab, tip = fuentes["B"][key]
            sc = scores_g0(fl, w)
            res = {}
            for c in (1, 2, 3):
                prs, prev = [], []
                for s in range(S):
                    t = tip[a:b, s]; y = (t == c).astype(int)
                    mantener = (lab[a:b, s] != -1) & ((t == 0) | (t == c)) & ~np.isnan(sc[a:b, s])
                    yy, ss = y[mantener], sc[a:b, s][mantener]
                    if yy.sum() == 0 or yy.min() == yy.max():
                        continue
                    prs.append(average_precision_score(yy, ss)); prev.append(yy.mean())
                res[NOMBRE_TIPO[c]] = {"pr_auc": float(np.mean(prs)), "prevalencia": float(np.mean(prev)),
                                       "lift": float(np.mean(prs) / np.mean(prev)), "n_bloques": len(prs)}
            por_tipo[cond][nombre] = res
    salida["B_pr_auc_por_tipo"] = por_tipo
    print("3) Composicion de instantes (prueba):", {k: round(v, 3) for k, v in comp["prueba"].items()})
    for tname, v in por_tipo[f"G0_w{w_star}"]["prueba"].items():
        print(f"   G0 w={w_star} prueba, tipo {tname:>10}: PR-AUC={v['pr_auc']:.3f} (lift {v['lift']:.1f}x, bloques {v['n_bloques']})")

    # ---------- 4) normalizacion bajo el Esquema B ----------
    norm = {}
    for pl in part["pliegues"]:
        k = pl["pliegue"]; t0, t1 = pl["train_idx"]
        z = np.load(P / f"esquema_B_pliegue{k}.npz")["data"][t0:t1]
        norm[f"pliegue{k}"] = {}
        for i, v in enumerate(["flow", "occupancy", "speed"]):
            mo, so = float(raw[t0:t1, :, i].mean()), float(raw[t0:t1, :, i].std())
            mb, sb = float(z[:, :, i].mean()), float(z[:, :, i].std())
            norm[f"pliegue{k}"][v] = {"mu_original": mo, "mu_B": mb, "dif_mu_rel_pct": 100 * (mb - mo) / mo,
                                      "sigma_original": so, "sigma_B": sb, "dif_sigma_rel_pct": 100 * (sb - so) / so}
    z = np.load(P / "esquema_B_final.npz")["data"][:te[0]]
    norm["final"] = {}
    for i, v in enumerate(["flow", "occupancy", "speed"]):
        mo, so = float(raw[:te[0], :, i].mean()), float(raw[:te[0], :, i].std())
        mb, sb = float(z[:, :, i].mean()), float(z[:, :, i].std())
        norm["final"][v] = {"mu_original": mo, "mu_B": mb, "dif_mu_rel_pct": 100 * (mb - mo) / mo,
                            "sigma_original": so, "sigma_B": sb, "dif_sigma_rel_pct": 100 * (sb - so) / so}
    salida["normalizacion_bajo_B"] = norm
    mx = max(abs(x[v][f]) for x in norm.values() for v in x for f in ("dif_mu_rel_pct", "dif_sigma_rel_pct"))
    print(f"4) Normalizacion: diferencia relativa maxima mu/sigma (B vs original) = {mx:.2f} %")
    for k, x in norm.items():
        print(f"   {k:>9} flow: mu {x['flow']['mu_original']:.2f} -> {x['flow']['mu_B']:.2f} ({x['flow']['dif_mu_rel_pct']:+.2f} %), "
              f"sigma {x['flow']['sigma_original']:.2f} -> {x['flow']['sigma_B']:.2f} ({x['flow']['dif_sigma_rel_pct']:+.2f} %)")

    (R / "g0_ampliado.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# G0 ampliado (Semana 6): bloques, recall 0.90, tipos de anomalia, normalizacion bajo B\n",
         "No cambia la seleccion de G0 (Decision 11). Reproduce `g0_resultados.json` (asserts).\n",
         "## Recall 0.90 (umbral fijado en la validacion del pliegue 3, aplicado a la prueba)\n",
         "| Esquema | Config. | Umbral | Recall prueba | FPR prueba | k por F1 solo en pliegue 3 | k Decision 11 |", "|---|---|---|---|---|---|---|"]
    for esq in ("A", "B"):
        for cond, o in salida["recall090"][esq].items():
            L.append(f"| {esq} | w={o['ventana']} ({'elegida' if cond == 'w_star' else 'sensibilidad'}) | {o['umbral_recall090_val_pliegue3']:.3f} | "
                     f"{o['recall_prueba']:.3f} | {o['fpr_prueba_a_ese_umbral']:.4f} | {o['k_por_F1_solo_pliegue3']} | {o['k_decision11_media_3_pliegues'] if o['k_decision11_media_3_pliegues'] is not None else '-'} |")
    L += ["\n## Esquema B: instantes anomalos por tipo (prueba)\n", "| Tipo | % de instantes |", "|---|---|"]
    L += [f"| {k} | {100 * v:.1f} |" for k, v in comp["prueba"].items()]
    L += ["\n## Esquema B: PR-AUC de G0 por tipo de anomalia (prueba; positivos = ese tipo, negativos = instantes normales)\n",
          "| Config. | Tipo | PR-AUC | Lift | Bloques |", "|---|---|---|---|---|"]
    for cond, d in por_tipo.items():
        for tname, v in d["prueba"].items():
            L.append(f"| {cond} | {tname} | {v['pr_auc']:.3f} | {v['lift']:.1f} | {v['n_bloques']} |")
    L += ["\n## Normalizacion bajo el Esquema B (flujo): serie original vs serie inyectada del train\n",
          "| Bloque | mu original | mu B | dif. % | sigma original | sigma B | dif. % |", "|---|---|---|---|---|---|---|"]
    for k, x in norm.items():
        f = x["flow"]
        L.append(f"| {k} | {f['mu_original']:.2f} | {f['mu_B']:.2f} | {f['dif_mu_rel_pct']:+.2f} | {f['sigma_original']:.2f} | {f['sigma_B']:.2f} | {f['dif_sigma_rel_pct']:+.2f} |")
    (R / "g0_ampliado.md").write_text("\n".join(L), encoding="utf-8")
    print("\nGuardado: results/eda/g0_por_bloque.csv, g0_ampliado.json, g0_ampliado.md")


if __name__ == "__main__":
    main()
