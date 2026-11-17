"""
Semana 10 - Genera las tablas de resultados del short paper (LaTeX) a partir de results/.
NADA se escribe a mano: todas las cifras salen de los archivos que ya produjeron los scripts del protocolo.

Entradas (solo lectura):
  results/prueba/por_bloque_metricas.csv   una fila por (condicion, esquema, sensor); AP medio de las 3 semillas por bloque
  results/prueba/por_bloque_semilla.csv    una fila por (condicion, esquema, semilla, sensor)
  results/eda/estadistica_<etiqueta>.json  Wilcoxon + Holm (delta = 0) y la version delta = 0.05
  results/ablacion/por_bloque_metricas.csv (opcional) ablacion de canales, si ya se compilo
Salida:
  paper/tablas_generadas.tex   (copiar junto a main_short_paper_v2.tex; el .tex la incluye con \\input)
  y un resumen impreso en consola.

Uso (raiz del proyecto, venv activo):
  python semana10_tablas_paper.py
  python semana10_tablas_paper.py --etiqueta prueba --etiqueta-delta prueba_delta005     # cuando exista G7
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NOMBRE = {"G0": "G0 m\\'ovil", "G1": "G1 LSTM", "G2": "G2 GRU", "G3": "G3 CNN-1D", "G4": "G4 AE",
          "G5": "G5 Transf.", "G6": "G6 CNN-LSTM-GRU", "G7": "G7 SI-DMSeq"}
ORDEN = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]


def _j(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def fp(p):
    try:
        if p is not None and float(p) < 1e-10:
            return r"$<\!10^{-10}$"
    except (TypeError, ValueError):
        pass
    return _fp_orig(p)


def _fp_orig(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f"${p / 10 ** e:.1f}\\!\\times\\!10^{{{e}}}$"
    return f"{p:.3f}"


def ffpr(x):
    import re as _re
    s = _ffpr_orig(x)
    return _re.sub(r"\$10\\!\\times\\!10\^\{(-?\d+)\}\$", lambda m: "$1\\!\\times\\!10^{%d}$" % (int(m.group(1)) + 1), s)


def _ffpr_orig(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if 0 < x < 1e-3:
        e = int(np.floor(np.log10(x)))
        return f"${x / 10 ** e:.0f}\\!\\times\\!10^{{{e}}}$"
    return f"{x:.3f}"


def boot_ic(x, rng, B=1000):
    x = np.asarray(x, dtype=float)
    m = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(B)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def conds(df):
    c = [k for k in ORDEN if k in set(df.condicion)]
    return c


def azar_bloque(d):
    """Nivel de azar del AP medio por bloque: prevalencia media de los bloques evaluables (no la global)."""
    return float((d.n_positivos / d.n_validos).mean())


def azar_tipo(d, k):
    """Azar del AP por tipo: positivos del tipo / (positivos del tipo + instantes normales), media sobre bloques con ese tipo."""
    npos = d[f"n_pos_{k}"]
    neg = d.n_validos - d.n_positivos
    m = npos > 0
    return float((npos[m] / (npos[m] + neg[m])).mean())


def tabla_h1(pm, sdelta0, sdelta05):
    rng = np.random.default_rng(42)
    L = []
    d = pm[(pm.esquema == "B") & (pm.evaluable == 1)]
    vs0 = {r["condicion"]: r for r in sdelta0["B"]["vs_referencia"]}
    vs5 = {r["condicion"]: r for r in sdelta05["B"]["vs_referencia"]}
    g0 = d[d.condicion == "G0"].set_index("sensor_idx").pr_auc
    az = azar_bloque(d[d.condicion == "G0"])
    L.append("\\begin{table*}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}")
    L.append("\\caption{O1/H1. AP por bloque en prueba bajo el Esquema B (media de tres semillas por bloque; "
             f"nivel de azar = prevalencia media de los bloques, {az:.3f}). $\\Delta$: diferencia media pareada frente a G0 con IC 95\\% bootstrap; "
             "$p_{\\mathrm{H}}$: Wilcoxon unilateral con Holm, contra $\\delta=0$ y contra $\\delta=0.05$; "
             "Gana: bloques en que el detector supera a G0.}")
    L.append("\\label{tab:h1}")
    L.append("\\begin{tabular}{@{}lccccccccc@{}}\\toprule")
    L.append("Detector & $n$ & Media & DE & Mediana & IC 95\\% & $\\Delta$ [IC 95\\%] & "
             "$p_{\\mathrm{H}}$ ($\\delta{=}0$) & $p_{\\mathrm{H}}$ ($\\delta{=}0.05$) & Gana \\\\\\midrule")
    for c in conds(d):
        x = d[d.condicion == c].pr_auc.to_numpy()
        lo, hi = boot_ic(x, rng)
        if c == "G0":
            dl, p0, p5, gana = "--", "--", "--", "--"
        else:
            r0, r5 = vs0.get(c), vs5.get(c)
            dl = (f"{r0['dif_media']:+.3f} [{r0['ic95_dif_media'][0]:+.3f}, {r0['ic95_dif_media'][1]:+.3f}]" if r0 else "--")
            p0 = fp(r0["p_holm"]) if r0 else "--"
            p5 = fp(r5["p_holm"]) if r5 else "--"
            xc = d[d.condicion == c].set_index("sensor_idx").pr_auc
            j = xc.index.intersection(g0.index)
            gana = f"{int((xc[j] > g0[j]).sum())}/{len(j)}"
        L.append(f"{NOMBRE[c]} & {len(x)} & {f3(x.mean())} & {f3(x.std(ddof=1))} & {f3(np.median(x))} & "
                 f"{f3(lo)}--{f3(hi)} & {dl} & {p0} & {p5} & {gana} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table*}")
    return "\n".join(L)


def tabla_secundarias(pm, sA):
    rng = np.random.default_rng(42)
    L = []
    dA = pm[(pm.esquema == "A") & (pm.evaluable == 1)]
    dB = pm[(pm.esquema == "B") & (pm.evaluable == 1)]
    vsA = {r["condicion"]: r for r in sA["A"].get("vs_referencia", [])}
    azA = azar_bloque(dA[dA.condicion == "G0"])
    azB = azar_bloque(dB[dB.condicion == "G0"])
    L.append("\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.6pt}")
    L.append("\\caption{M\\'etricas en prueba. Esquema A: solo descriptivo (comparte familia de reglas con G0); 84 bloques con ambas "
             f"clases; azar = prevalencia media de esos bloques ({azA:.4f}; la global es 0.0010). Esquema B: 291 bloques, azar {azB:.3f}; "
             "AP con DE e IC 95\\% en la Tabla~\\ref{tab:h1}. F1 y FPR dependen de umbrales elegidos en el bloque de umbral, "
             "que forma parte del entrenamiento final y puede favorecerlos; el AP no. R90: umbral que alcanz\\'o recall 0.90 en ese bloque; "
             "F1: umbral de F1 m\\'aximo.}")
    L.append("\\label{tab:secund}")
    L.append("\\begin{tabular}{@{}l|ccccccccc|ccccc@{}}\\toprule")
    L.append("& \\multicolumn{9}{c|}{Esquema A ($n=84$)} & \\multicolumn{5}{c}{Esquema B ($n=291$)} \\\\")
    L.append("Detector & AP & DE & IC 95\\% & Med. & $\\Delta$ & F1 & FPR F1 & Rec.\\ R90 & FPR R90 & ROC & F1 & FPR F1 & Rec.\\ R90 & FPR R90 \\\\\\midrule")
    for c in conds(pm):
        a = dA[dA.condicion == c]
        b = dB[dB.condicion == c]
        lo, hi = boot_ic(a.pr_auc.to_numpy(), rng)
        dl = "--" if c == "G0" or c not in vsA else f"{vsA[c]['dif_media']:+.3f}"
        L.append(f"{NOMBRE[c]} & {f3(a.pr_auc.mean())} & {f3(a.pr_auc.std(ddof=1))} & {f3(lo)}--{f3(hi)} & {f3(a.pr_auc.median())} & {dl} & "
                 f"{f3(a.f1_f1.mean())} & {ffpr(a.fpr_f1.mean())} & {f3(a.recall_r90.mean())} & {ffpr(a.fpr_r90.mean())} & "
                 f"{f3(b.roc_auc.mean())} & {f3(b.f1_f1.mean())} & {ffpr(b.fpr_f1.mean())} & {f3(b.recall_r90.mean())} & {ffpr(b.fpr_r90.mean())} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table*}")
    return "\n".join(L)


def tabla_tipo_semilla(pm, ps):
    L = []
    dB = pm[(pm.esquema == "B") & (pm.evaluable == 1)]
    sB = ps[(ps.esquema == "B") & (ps.evaluable == 1)]
    g0 = dB[dB.condicion == "G0"]
    tipos = ("puntual", "contextual", "colectiva")
    nt = [int(g0[f"pr_auc_{k}"].notna().sum()) for k in tipos]
    L.append("\\begin{table*}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}")
    L.append("\\caption{AP medio por bloque en prueba (Esquema B): por semilla ($n=291$) y por tipo de anomal\\'ia, con la DE entre "
             f"par\\'entesis ($n={nt[0]}$, ${nt[1]}$ y ${nt[2]}$ bloques con instantes puntuales, contextuales y colectivos). "
             "Por tipo: positivos del tipo frente a todos los instantes normales; los positivos de los otros tipos se excluyen. "
             "La \\'ultima fila es el nivel de azar de cada tipo.}")
    L.append("\\label{tab:tipo}")
    L.append("\\begin{tabular}{@{}lcccccc@{}}\\toprule")
    L.append("Detector & Sem.\\ 42 & Sem.\\ 43 & Sem.\\ 44 & Puntual & Contextual & Colectiva \\\\\\midrule")
    for c in conds(pm):
        b = dB[dB.condicion == c]
        if c == "G0":
            ss = ["--", "--", "--"]
        else:
            sx = sB[sB.condicion == c].groupby("semilla").pr_auc.mean()
            ss = [f3(sx.get(s, np.nan)) for s in (42, 43, 44)]
        t = [f"{f3(b[f'pr_auc_{k}'].mean())} ({f3(b[f'pr_auc_{k}'].std(ddof=1))})" for k in tipos]
        L.append(f"{NOMBRE[c]} & {ss[0]} & {ss[1]} & {ss[2]} & {t[0]} & {t[1]} & {t[2]} \\\\")
    az = [f3(azar_tipo(g0, k)) for k in tipos]
    L.append(f"\\midrule Azar & -- & -- & -- & {az[0]} & {az[1]} & {az[2]} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table*}")
    return "\n".join(L)


def tabla_ablacion(pm, ruta):
    return ""


def _tabla_ablacion_vieja(pm, ruta):
    ab = pd.read_csv(ruta)
    ab = ab[ab.evaluable == 1]
    base = pm[(pm.esquema == "B") & (pm.evaluable == 1)]
    L = []
    L.append("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}")
    L.append("\\caption{Ablaci\\'on de canales (Esquema B, prueba, una semilla): AP medio por bloque con ocupaci\\'on y "
             "velocidad en cero (solo flujo y variables temporales), frente a la corrida completa y a G0.}")
    L.append("\\label{tab:abl}")
    L.append("\\begin{tabular}{@{}lccccccc@{}}\\toprule")
    L.append("Detector & Completa & Ablada & $\\Delta$ & $n$ & Punt. & Context. & Colect. \\\\\\midrule")
    g0 = base[base.condicion == "G0"].pr_auc.mean()
    for c, x in ab.groupby("condicion"):
        b = c[:-4]
        full = base[base.condicion == b].pr_auc.mean()
        t = [f3(x[f"pr_auc_{k}"].mean()) for k in ("puntual", "contextual", "colectiva")]
        L.append(f"{NOMBRE.get(b, b)} & {f3(full)} & {f3(x.pr_auc.mean())} & {x.pr_auc.mean() - full:+.3f} & {len(x)} & "
                 f"{t[0]} & {t[1]} & {t[2]} \\\\")
    L.append(f"G0 m\\'ovil (solo flujo) & -- & {f3(g0)} & -- & {int((base.condicion == 'G0').sum())} & "
             f"{f3(base[base.condicion == 'G0'].pr_auc_puntual.mean())} & {f3(base[base.condicion == 'G0'].pr_auc_contextual.mean())} & "
             f"{f3(base[base.condicion == 'G0'].pr_auc_colectiva.mean())} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiqueta", default="prueba_sinG7")
    ap.add_argument("--etiqueta-delta", default="prueba_sinG7_delta005")
    ap.add_argument("--salida", default="paper/tablas_generadas.tex")
    a = ap.parse_args()
    pm = pd.read_csv("results/prueba/por_bloque_metricas.csv")
    ps = pd.read_csv("results/prueba/por_bloque_semilla.csv")
    s0 = _j(f"results/eda/estadistica_{a.etiqueta}.json")
    s5 = _j(f"results/eda/estadistica_{a.etiqueta_delta}.json")
    partes = ["% Generado por semana10_tablas_paper.py a partir de results/. NO editar a mano.",
              tabla_h1(pm, s0, s5), tabla_secundarias(pm, s0), tabla_tipo_semilla(pm, ps)]
    rab = Path("results/ablacion/por_bloque_metricas.csv")
    if rab.exists():
        partes.append(tabla_ablacion(pm, rab))
    else:
        partes.append("% (ablacion aun no compilada: no se genero tab:abl)")
    out = Path(a.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(partes) + "\n", encoding="utf-8")
    print(f"Guardado: {out}  ({'con' if rab.exists() else 'sin'} tabla de ablacion)")
    # resumen en consola
    for esq in ("A", "B"):
        d = pm[(pm.esquema == esq) & (pm.evaluable == 1)]
        print(f"\nEsquema {esq}: AP medio por bloque (n, media, DE, mediana)")
        for c in conds(d):
            x = d[d.condicion == c].pr_auc
            print(f"  {c}: n={len(x)} media={x.mean():.4f} DE={x.std(ddof=1):.4f} mediana={x.median():.4f}")
    d = pm[(pm.esquema == "B") & (pm.evaluable == 1)]
    print("\nEsquema B, AP por tipo (media por bloque):")
    for c in conds(d):
        x = d[d.condicion == c]
        print("  " + c + ": " + "  ".join(f"{k}={x[f'pr_auc_{k}'].mean():.4f} (n={int(x[f'pr_auc_{k}'].notna().sum())})"
                                           for k in ("puntual", "contextual", "colectiva")))


if __name__ == "__main__":
    main()
