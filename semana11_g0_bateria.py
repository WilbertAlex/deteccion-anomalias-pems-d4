"""
Semana 11 - Bateria de reglas moviles y G0 multicanal (revision del asesor, obs. 1 y 3). POSTERIOR Y EXPLORATORIO.

Reglas no entrenadas (sin etiquetas), todas con puntaje "cuanto se aleja el instante t de su pasado" (sin mirar t en la
referencia, shift(1)). Su ventana/parametro se elige por AP medio por sensor en el bloque de validacion 'umbral' (pliegue 3),
igual que G0; luego se evaluan UNA vez en la prueba. El AP no depende del umbral k, solo del orden de los puntajes.
  R1 media/desv. movil (flujo)           = G0 sellado (referencia; w del config)
  R2 mediana/MAD movil (flujo)
  R3 EWMA media/desv. (flujo)
  R4 perfil horario (flujo): media y desv. por franja de 5 min y tipo de dia (laboral/fin de semana) del bloque de 80 %
  R5 diferencia de primer orden (flujo): z movil de x_t - x_(t-1)
  R6 G0 multicanal: maximo del z movil de flujo, ocupacion y velocidad (media/desv. movil)
  R7 G0 multicanal robusto: maximo del z (mediana/MAD) de los tres canales
Uso:  python semana11_g0_bateria.py            (niveles de dificultad 1 0.5 0.25 por defecto)
Escribe results/g0_bateria/bateria.csv
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, pandas as pd

P = Path("data/processed"); OUT = Path("results/g0_bateria")
VENT = (12, 22, 48, 96, 288)
EPS = 1e-9


def zmov(X, w, robusto=False):
    df = pd.DataFrame(X)
    if robusto:
        med = df.rolling(w, min_periods=w).median().shift(1)
        dev = (df - med).abs()
        mad = dev.rolling(w, min_periods=w).median().shift(1) * 1.4826
        z = ((df - med).abs() / mad.where(mad > EPS)).to_numpy()
        z = np.where(mad.to_numpy() <= EPS, 0.0, z)
    else:
        mu = df.rolling(w, min_periods=w).mean().shift(1); sd = df.rolling(w, min_periods=w).std(ddof=0).shift(1)
        z = ((df - mu).abs() / sd.where(sd > EPS)).to_numpy()
        z = np.where(sd.to_numpy() <= EPS, 0.0, z)
    return np.where(np.isnan(z), np.nan, z)


def zewma(X, w):
    df = pd.DataFrame(X)
    mu = df.ewm(span=w, adjust=False).mean().shift(1); sd = df.ewm(span=w, adjust=False).std().shift(1)
    z = ((df - mu).abs() / sd.where(sd > EPS)).to_numpy()
    z = np.where(sd.to_numpy() <= EPS, 0.0, z)
    return np.where(np.isnan(z), np.nan, z)


def zperfil(X, tr1):
    T = X.shape[0]
    franja = np.arange(T) % 288
    dia = ((np.arange(T) // 288) % 7)                       # tipo de dia por posicion; laboral/fin de semana segun PeMS (abajo)
    tipo = (dia >= 5).astype(int)                           # 0..4 = lun-vie si la serie empieza en lunes (PeMS-D4: 1 ene 2018 = lunes)
    mu = np.zeros((2, 288, X.shape[1])); sd = np.ones((2, 288, X.shape[1]))
    for t in (0, 1):
        for fr in range(288):
            m = (franja[:tr1] == fr) & (tipo[:tr1] == t)
            if m.sum() >= 3:
                mu[t, fr] = X[:tr1][m].mean(0); sd[t, fr] = np.maximum(X[:tr1][m].std(0), 1e-6)
    return np.abs(X - mu[tipo, franja]) / sd[tipo, franja]


def reglas(flujo, ocu, vel, tr1):
    """{nombre: {w: funcion sin argumentos que devuelve el puntaje (T,S)}}"""
    dflujo = np.vstack([np.full((1, flujo.shape[1]), np.nan), np.diff(flujo, axis=0)])
    R = {"R2 mediana/MAD": {}, "R3 EWMA": {}, "R5 diferencia": {}, "R6 multicanal": {}, "R7 multicanal robusto": {}}
    for w in VENT:
        R["R2 mediana/MAD"][w] = lambda w=w: zmov(flujo, w, True)
        R["R3 EWMA"][w] = lambda w=w: zewma(flujo, w)
        R["R5 diferencia"][w] = lambda w=w: zmov(dflujo, w)
        R["R6 multicanal"][w] = lambda w=w: np.fmax(np.fmax(zmov(flujo, w), zmov(ocu, w)), zmov(vel, w))
        R["R7 multicanal robusto"][w] = lambda w=w: np.fmax(np.fmax(zmov(flujo, w, True), zmov(ocu, w, True)), zmov(vel, w, True))
    R["R4 perfil horario"] = {0: lambda: zperfil(flujo, tr1)}
    return R


def ap_de(sc, lab, tipos, a, b, P10):
    df = P10.metricas_bloques(sc[a:b], lab[a:b], np.inf, np.inf, tipos[a:b])
    return df, df[df.evaluable == 1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--niveles", nargs="+", type=float, default=[1.0, 0.5, 0.25])
    a = ap.parse_args()
    import semana10_prueba_final as P10
    from g0_linea_base import scores_g0
    base = np.load(P / "pems04_semana2.npz")["data"].astype(np.float64)
    z = np.load(P / "esquema_B_final.npz"); alt = z["data"].astype(np.float64); lab = z["labels"]
    tipos = P10.tipos_b(); Rg = P10._rangos_split(); (va0, va1), (te0, te1) = Rg["umbral"], Rg["test"]
    tr1 = 13593
    w0 = int(P10._j("configs/config_g0.json")["seleccion_observada"]["esquema_B"]["ventana"])
    # 1) seleccion de w por AP en validacion (Esquema B original, d = 1), una sola vez por regla
    elegido = {}
    Rv = reglas(alt[:, :, 0], alt[:, :, 1], alt[:, :, 2], tr1)
    for nom, lista in Rv.items():
        mejor = max(lista, key=lambda w: ap_de(lista[w](), lab, tipos, va0, va1, P10)[1].pr_auc.mean())
        elegido[nom] = mejor; print(f"{nom}: w elegido por validacion = {mejor}", flush=True)
    # 2) evaluacion en la prueba por nivel de dificultad
    filas = []
    for d in a.niveles:
        X = base + d * (alt - base)
        Rd = reglas(X[:, :, 0], X[:, :, 1], X[:, :, 2], tr1)
        Rd["R1 G0 sellado"] = {w0: lambda: scores_g0(X[:, :, 0], w0)}; elegido["R1 G0 sellado"] = w0
        for nom, lista in Rd.items():
            w = elegido[nom]; df, ev = ap_de(lista[w](), lab, tipos, te0, te1, P10)
            fila = {"regla": nom, "w": w, "nivel": d, "ap_prueba": ev.pr_auc.mean(), "n": len(ev),
                    "ap_puntual": df.pr_auc_puntual.mean(), "ap_contextual": df.pr_auc_contextual.mean(), "ap_colectiva": df.pr_auc_colectiva.mean()}
            filas.append(fila); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()}, flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            df.assign(regla=nom, nivel=d).to_csv(OUT / f"por_sensor_{nom.split()[0]}_d{d}.csv", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(filas); df.to_csv(OUT / "bateria.csv", index=False)
    print(df.pivot(index="regla", columns="nivel", values="ap_prueba").round(3))
    # 3) redes supervisadas (prueba, Esquema B, d = 1) frente a las dos mejores reglas no entrenadas: Wilcoxon unilateral + Holm
    try:
        from scipy.stats import wilcoxon
        pb = pd.read_csv("results/prueba/por_bloque.csv")
        pb = pb[(pb.esquema == "B") & (pb.split == "prueba") & (pb.evaluable == 1)]
        W = pb.pivot_table(index="sensor_idx", columns="condicion", values="pr_auc")
        mejores = df[df.nivel == 1.0].sort_values("ap_prueba", ascending=False).regla.head(2).tolist()
        filas2 = []
        for r in mejores:
            ref = pd.read_csv(OUT / f"por_sensor_{r.split()[0]}_d1.0.csv").set_index("sensor_idx").pr_auc
            for c in [x for x in W.columns if x not in ("G0", "G4")]:
                x = (W[c] - ref).dropna()
                filas2.append({"regla": r, "red": c, "dif_media": x.mean(), "p": wilcoxon(x, alternative="greater").pvalue})
        t = pd.DataFrame(filas2).sort_values("p"); m = len(t)
        t["p_holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * t.p.to_numpy()))
        print("\nRedes frente a las mejores reglas no entrenadas (d = 1):"); print(t.round(4).to_string(index=False))
        t.to_csv(OUT / "redes_vs_mejores_reglas.csv", index=False)
    except Exception as e:
        print("No se pudo comparar con las redes:", e)


if __name__ == "__main__":
    main()
