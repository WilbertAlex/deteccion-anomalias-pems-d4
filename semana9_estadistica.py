"""
Semana 9 - Script de analisis estadistico (PPI sec. 1.6 y 2.9; hallazgos H-05 y H-06 de la revision del 2 nov).

Entrada: tabla larga POR BLOQUE (sensor x pliegue), con el formato de results/eda/g0_por_bloque.csv:
  condicion, esquema, split, sensor_idx, pr_auc, n_validos, n_positivos, evaluable
  - esquema: A | B ; split: val_pliegue1..3 | prueba ; evaluable: 1 si el bloque tiene ambas clases.
  - Cada condicion nueva (G1..G7) se agrega como mas filas con el mismo formato.

Calcula (segun el PPI):
  1. Friedman sobre los bloques evaluables comunes a todas las condiciones, rangos promedio y W de Kendall.
  2. Contraste confirmatorio frente a la referencia (G0): prueba pareada de Wilcoxon sobre la diferencia de
     PR-AUC por bloque y correccion de Holm; IC 95 % bootstrap de la diferencia media.
     Si --delta > 0 se contrasta H1 con margen: diferencia > delta (Wilcoxon sobre d - delta).
  3. Nemenyi (exploratorio, no confirmatorio): valor p por par y distancia critica.
  4. H3: tau de Kendall (y rho de Spearman como sensibilidad) entre los rankings de PR-AUC de A y B sobre los
     bloques comunes a ambos esquemas, con IC 95 % bootstrap remuestreando bloques (B = 1000).
     Lectura segun el PPI: estable si tau >= 0.7; concluyente solo si el IC completo queda a un lado de 0.7.
  5. Analisis complementario de H-06: tau entre A y B con PR-AUC AGREGADO por pliegue (--agregado).

ELECCIONES PENDIENTES DEL EQUIPO (el script las deja como parametros, NO las decide):
  --conjunto val|prueba   conjunto sobre el que se contrasta H1 (el PPI: validacion; el short paper: prueba).
  --delta                 margen minimo de mejora (0.05 propuesto en el short paper; 0.0 = sin margen).
  --alternativa           greater (H1 de mejora, por defecto) | two-sided.
La prueba de significancia usa los bloques como unidades; los bloques de un mismo sensor en distintos pliegues
y los sensores vecinos no son independientes. Es una limitacion del diseno del PPI y se declara al reportar.

Uso (desde la raiz del proyecto):
  python semana9_estadistica.py --autoprueba                                   # prueba con datos simulados
  python semana9_estadistica.py --etiqueta g0 --entrada results/eda/g0_por_bloque.csv --conjunto val
Genera: results/eda/estadistica_<etiqueta>.json y .md (no sobrescribe).
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALFA = 0.05
B_BOOT = 1000  # PPI sec. 2.9


# ---------------------------------------------------------------- utilidades
def holm(p):
    """Correccion de Holm-Bonferroni (valores p ajustados, monotonos)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    aj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        aj[j] = min(1.0, run)
    return aj


def rangos_promedio(M):
    """Rango 1 = mejor (mayor PR-AUC). M: (N bloques, k condiciones). Empates: rango medio."""
    r = np.apply_along_axis(lambda f: stats.rankdata(-f), 1, M)
    return r.mean(axis=0)


def tabla_bloques(df, esquema, conjunto):
    """Matriz bloques x condiciones con los bloques evaluables en TODAS las condiciones."""
    d = df[(df.esquema == esquema) & (df.evaluable == 1)].copy()
    d = d[d.split.str.startswith("val")] if conjunto == "val" else d[d.split == "prueba"]
    d["bloque"] = d.split + "|" + d.sensor_idx.astype(str)
    M = d.pivot_table(index="bloque", columns="condicion", values="pr_auc", aggfunc="first")
    return M.dropna(axis=0, how="any")


# ---------------------------------------------------------------- pruebas
def friedman(M):
    N, k = M.shape
    if k < 3:
        return {"aplicable": False, "motivo": f"Friedman exige >= 3 condiciones; hay {k}", "N": N, "k": k}
    chi2, p = stats.friedmanchisquare(*[M.iloc[:, j].to_numpy() for j in range(k)])
    return {"aplicable": True, "N": N, "k": k, "chi2": float(chi2), "p": float(p),
            "W_kendall": float(chi2 / (N * (k - 1))),
            "rangos_promedio": dict(zip(M.columns, map(float, rangos_promedio(M.to_numpy()))))}


def contraste_vs_ref(M, ref, delta, alternativa, rng):
    """Wilcoxon pareado de cada condicion contra la referencia, con Holm y IC bootstrap de la diferencia."""
    filas = []
    for c in M.columns:
        if c == ref:
            continue
        d = (M[c] - M[ref]).to_numpy()
        z = d - delta
        p = 1.0 if np.all(z == 0) else float(stats.wilcoxon(z, alternative=alternativa).pvalue)
        idx = rng.integers(0, len(d), size=(B_BOOT, len(d)))
        medias = d[idx].mean(axis=1)
        filas.append({"condicion": c, "n_bloques": int(len(d)), "dif_media": float(d.mean()),
                      "dif_mediana": float(np.median(d)),
                      "ic95_dif_media": [float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))],
                      "p": p, "prop_bloques_mejor": float((d > 0).mean())})
    if filas:
        for f, pa in zip(filas, holm([f["p"] for f in filas])):
            f["p_holm"] = float(pa)
            f["rechaza_H0_holm"] = bool(pa < ALFA)
    return filas


def nemenyi(M):
    N, k = M.shape
    if k < 3:
        return {"aplicable": False}
    r = rangos_promedio(M.to_numpy())
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    cd = stats.studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2) * se
    pares = []
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(r[i] - r[j]) / se * np.sqrt(2)
            pares.append({"a": M.columns[i], "b": M.columns[j], "dif_rangos": float(abs(r[i] - r[j])),
                          "p": float(1 - stats.studentized_range.cdf(q, k, np.inf))})
    return {"aplicable": True, "distancia_critica": float(cd), "pares": pares}


def tau_rho(a, b):
    return float(stats.kendalltau(a, b).statistic), float(stats.spearmanr(a, b).statistic)


def estabilidad(df, conjunto, rng):
    """H3: tau y rho entre rankings (media de PR-AUC por condicion) de A y B, con IC bootstrap sobre bloques."""
    MA, MB = tabla_bloques(df, "A", conjunto), tabla_bloques(df, "B", conjunto)
    comunes = MA.index.intersection(MB.index)
    conds = [c for c in MA.columns if c in MB.columns]
    if len(conds) < 3 or len(comunes) < 10:
        return {"aplicable": False, "motivo": f"{len(conds)} condiciones y {len(comunes)} bloques comunes"}
    A, Bm = MA.loc[comunes, conds].to_numpy(), MB.loc[comunes, conds].to_numpy()
    tau, rho = tau_rho(A.mean(0), Bm.mean(0))
    taus, rhos = [], []
    for _ in range(B_BOOT):
        i = rng.integers(0, len(comunes), len(comunes))
        t, r = tau_rho(A[i].mean(0), Bm[i].mean(0))
        if not (np.isnan(t) or np.isnan(r)):
            taus.append(t); rhos.append(r)
    ic = [float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))]
    lectura = ("estable (IC completo >= 0.7)" if ic[0] >= 0.7 else
               "sensible al esquema (IC completo < 0.7)" if ic[1] < 0.7 else "no concluyente (IC cruza 0.7)")
    return {"aplicable": True, "n_bloques_comunes": int(len(comunes)), "condiciones": conds, "tau": tau,
            "ic95_tau": ic, "rho_spearman": rho,
            "ic95_rho": [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
            "lectura_ppi": lectura, "punto_tau>=0.7": bool(tau >= 0.7),
            "ranking_A": [conds[i] for i in np.argsort(-A.mean(0))],
            "ranking_B": [conds[i] for i in np.argsort(-Bm.mean(0))]}


def complementario_agregado(ruta):
    """H-06: tau entre A y B con PR-AUC agregado por pliegue. CSV: condicion,esquema,split,pr_auc_agregado
    o el JSON de semana8_g0_agregado.py ({A|B: {condicion: {split: {pr_auc_agregado}}}})."""
    ruta = Path(ruta)
    if ruta.suffix == ".json":
        j = json.loads(ruta.read_text(encoding="utf-8"))
        filas = [(c, e, s, v["pr_auc_agregado"]) for e in ("A", "B") for c, d in j[e].items() for s, v in d.items()]
        g = pd.DataFrame(filas, columns=["condicion", "esquema", "split", "pr_auc_agregado"])
    else:
        g = pd.read_csv(ruta)
    out = {}
    for s in sorted(g.split.unique()):
        a = g[(g.split == s) & (g.esquema == "A")].set_index("condicion").pr_auc_agregado
        b = g[(g.split == s) & (g.esquema == "B")].set_index("condicion").pr_auc_agregado
        cs = [c for c in a.index if c in b.index and not (np.isnan(a[c]) or np.isnan(b[c]))]
        if len(cs) < 3:
            out[s] = {"aplicable": False, "motivo": f"{len(cs)} condiciones comparables"}
        else:
            t, r = tau_rho(a[cs].to_numpy(), b[cs].to_numpy())
            out[s] = {"aplicable": True, "condiciones": cs, "tau": t, "rho": r}
    return out


# ---------------------------------------------------------------- informe
def analizar(df, ref, conjunto, delta, alternativa, semilla, agregado=None):
    rng = np.random.default_rng(semilla)
    res = {"parametros": {"referencia": ref, "conjunto": conjunto, "delta": delta, "alternativa": alternativa,
                          "alfa": ALFA, "B_bootstrap": B_BOOT, "semilla": semilla}}
    for esq in ("A", "B"):
        M = tabla_bloques(df, esq, conjunto)
        todos = df[(df.esquema == esq) & (df.split.str.startswith("val") if conjunto == "val" else df.split == "prueba")]
        n_tot = int(todos.groupby(["split", "sensor_idx"]).ngroups)
        r = {"bloques_totales": n_tot, "bloques_evaluables_comunes": int(len(M)), "condiciones": list(M.columns)}
        if ref in M.columns and len(M) > 0:
            r["friedman"] = friedman(M)
            r["vs_referencia"] = contraste_vs_ref(M, ref, delta, alternativa, rng)
            r["nemenyi_exploratorio"] = nemenyi(M)
        else:
            r["error"] = f"referencia {ref} ausente o sin bloques evaluables comunes"
        res[esq] = r
    res["H3_estabilidad"] = estabilidad(df, conjunto, rng)
    if agregado:
        res["H3_complementario_agregado_por_pliegue"] = complementario_agregado(agregado)
    return res


def a_markdown(res):
    L = ["# Analisis estadistico (salida de semana9_estadistica.py)", "", f"Parametros: `{json.dumps(res['parametros'])}`", ""]
    for esq in ("A", "B"):
        r = res[esq]
        L += [f"## Esquema {esq}", f"Bloques: {r['bloques_evaluables_comunes']} evaluables comunes de {r['bloques_totales']}. "
              f"Condiciones: {', '.join(r['condiciones'])}.", ""]
        if "error" in r:
            L += [r["error"], ""]; continue
        f = r["friedman"]
        L.append("Friedman: " + (f"chi2 = {f['chi2']:.2f}, p = {f['p']:.3g}, W = {f['W_kendall']:.3f}" if f["aplicable"] else f["motivo"]))
        L += ["", "| Condicion | n | dif. media | IC95 % | p | p Holm | rechaza H0 |", "|---|---|---|---|---|---|---|"]
        for v in r["vs_referencia"]:
            L.append(f"| {v['condicion']} | {v['n_bloques']} | {v['dif_media']:+.4f} | [{v['ic95_dif_media'][0]:+.4f}, "
                     f"{v['ic95_dif_media'][1]:+.4f}] | {v['p']:.3g} | {v['p_holm']:.3g} | {'si' if v['rechaza_H0_holm'] else 'no'} |")
        L.append("")
    h = res["H3_estabilidad"]
    L.append("## H3 (estabilidad del ranking)")
    if h["aplicable"]:
        L += [f"tau = {h['tau']:.3f}, IC95 % [{h['ic95_tau'][0]:.3f}, {h['ic95_tau'][1]:.3f}]; rho = {h['rho_spearman']:.3f}. "
              f"Lectura segun el PPI: {h['lectura_ppi']}. Bloques comunes: {h['n_bloques_comunes']}.",
              f"Ranking A: {' > '.join(h['ranking_A'])}", f"Ranking B: {' > '.join(h['ranking_B'])}"]
    else:
        L.append("No aplicable: " + h["motivo"])
    if "H3_complementario_agregado_por_pliegue" in res:
        L += ["", "Complementario (PR-AUC agregado por pliegue):", "```", json.dumps(res["H3_complementario_agregado_por_pliegue"], indent=1), "```"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- simulacion y autoprueba
def simular(efecto_B, efecto_A, rank_invertido=False, seed=0, n_sens=291, n_ev_A=0.30):
    """Tabla larga simulada: G0 + G1..G7 en A y B, 3 pliegues de validacion + prueba. Efectos conocidos."""
    rng = np.random.default_rng(seed)
    conds = ["G0"] + [f"G{i}" for i in range(1, 8)]
    filas = []
    for esq, ef in (("B", efecto_B), ("A", efecto_A)):
        for s in ["val_pliegue1", "val_pliegue2", "val_pliegue3", "prueba"]:
            sens = np.arange(n_sens)
            ev = np.ones(n_sens, bool) if esq == "B" else (np.random.default_rng(zlib.crc32(s.encode())).random(n_sens) < n_ev_A)
            base = rng.beta(2, 6, n_sens)  # dificultad propia del bloque (comun a las condiciones)
            for k, c in enumerate(conds):
                e = ef[k] if not (rank_invertido and esq == "A") else ef[len(conds) - 1 - k]
                pr = np.clip(base + e + rng.normal(0, 0.03, n_sens), 0, 1)
                for si in sens:
                    filas.append((c, esq, s, si, pr[si] if ev[si] else np.nan, 100, 1, int(ev[si])))
    return pd.DataFrame(filas, columns=["condicion", "esquema", "split", "sensor_idx", "pr_auc", "n_validos", "n_positivos", "evaluable"])


def autoprueba():
    """Comprueba el pipeline con escenarios de respuesta conocida."""
    nulo = [0.0] * 8
    efecto = [0.0, 0.02, 0.06, 0.10, 0.14, 0.18, 0.22, 0.26]
    # 1. Nulo: sin diferencias reales. No debe rechazar tras Holm (salvo falsos positivos raros).
    r = analizar(simular(nulo, nulo, seed=1), "G0", "val", 0.0, "greater", 1)
    rech = sum(v["rechaza_H0_holm"] for v in r["B"]["vs_referencia"])
    assert rech <= 1, f"escenario nulo rechazo {rech} de 7"
    assert r["B"]["friedman"]["p"] > 0.001
    # 2. Efecto creciente en B: G1 (+0.02) sin margen se detecta; con delta=0.05 solo G3..G7 (+0.10 en adelante... G2=+0.06)
    r = analizar(simular(efecto, nulo, seed=2), "G0", "val", 0.0, "greater", 2)
    assert r["B"]["friedman"]["p"] < 1e-6 and r["B"]["friedman"]["W_kendall"] > 0.3
    assert all(v["rechaza_H0_holm"] for v in r["B"]["vs_referencia"]), "debe detectar todos los efectos en B sin margen"
    r5 = analizar(simular(efecto, nulo, seed=2), "G0", "val", 0.05, "greater", 2)
    det = {v["condicion"]: v["rechaza_H0_holm"] for v in r5["B"]["vs_referencia"]}
    assert not det["G1"] and det["G3"] and det["G7"], det   # G1 (+0.02) no supera delta=0.05; G3 (+0.10) si
    # 3. H3: mismo orden en A y B -> estable; orden invertido -> sensible
    estable = estabilidad(simular(efecto, efecto, seed=3), "val", np.random.default_rng(3))
    assert estable["aplicable"] and estable["tau"] > 0.9 and estable["lectura_ppi"].startswith("estable"), estable
    inv = estabilidad(simular(efecto, efecto, rank_invertido=True, seed=3), "val", np.random.default_rng(3))
    assert inv["tau"] < -0.9 and inv["lectura_ppi"].startswith("sensible"), inv
    # 4. Nemenyi: distancia critica comparable a la tabla de Demsar (k=8, N grande -> CD pequena)
    nm = r["B"]["nemenyi_exploratorio"]
    assert nm["aplicable"] and nm["distancia_critica"] < 0.5
    # 5. Holm: ejemplo conocido
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    print("AUTOPRUEBA OK: nulo sin rechazos; efectos detectados; delta=0.05 filtra G1; H3 estable e invertido correctos; Holm correcto.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoprueba", action="store_true")
    ap.add_argument("--etiqueta")
    ap.add_argument("--entrada", default="results/eda/g0_por_bloque.csv")
    ap.add_argument("--referencia", default="G0_w22")
    ap.add_argument("--conjunto", choices=["val", "prueba"], default="val")
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--alternativa", choices=["greater", "two-sided"], default="greater")
    ap.add_argument("--agregado", default=None, help="CSV o results/eda/g0_agregado.json (complementario H-06)")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()
    if a.autoprueba:
        return autoprueba()
    if not a.etiqueta:
        sys.exit("Falta --etiqueta (o usa --autoprueba).")
    out = Path(f"results/eda/estadistica_{a.etiqueta}.json")
    if out.exists():
        sys.exit(f"Ya existe {out}. Usa otra --etiqueta para no sobrescribirlo.")
    df = pd.read_csv(a.entrada)
    res = analizar(df, a.referencia, a.conjunto, a.delta, a.alternativa, a.semilla, a.agregado)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".md").write_text(a_markdown(res), encoding="utf-8")
    print(a_markdown(res))
    print(f"Guardado: {out} y {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
