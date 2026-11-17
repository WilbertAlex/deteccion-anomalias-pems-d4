"""
Semana 11 - Gradiente de dificultad del Esquema B (revision del asesor, obs. 2). ANALISIS POSTERIOR Y EXPLORATORIO.

Idea: el Esquema B de la prueba se "suaviza" SIN tocar el generador sellado. Para cada nivel d en (1, 0.75, 0.5, 0.35, 0.25, 0.15):
    datos_d = original + d * (datos_B - original)        (solo en el bloque de prueba, solo cambia donde hay anomalia)
Las posiciones, los tipos y las etiquetas son identicos; solo se reduce la magnitud en TODOS los canales (flujo, ocupacion,
velocidad). d = 1 es el Esquema B original; d pequeno = anomalias casi indistinguibles. Los modelos NO se reentrenan:
es una prueba de robustez/transferencia a anomalias mas debiles que las vistas en entrenamiento.
Condiciones: G0 (sellado, w=22), C1..C4 (LR y GBM, se entrenan en 1 min con el entrenamiento estandar) y, con --redes, las
redes supervisadas con sus pesos sellados (semilla 42).

Uso (raiz del proyecto, venv activo):
  python semana11_dificultad.py --humo
  python semana11_dificultad.py                                     # G0 + LR + GBM, 6 niveles (minutos)
  python semana11_dificultad.py --redes lstm gru cnn1d transformer cnn_lstm_gru --niveles 1 0.5 0.25   # horas; se puede repartir
Escribe results/dificultad/ap_por_nivel_<etiqueta>.csv (una fila por condicion x nivel, con AP por tipo).
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, pandas as pd

OUT = Path("results/dificultad"); P = Path("data/processed")
NIVELES = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15)
COND_RED = {"lstm": "G1", "gru": "G2", "cnn1d": "G3", "transformer": "G5", "cnn_lstm_gru": "G6"}


def resumen(cond, d, M, L, T, P10):
    df = P10.metricas_bloques(M, L, np.inf, np.inf, T)
    ev = df[df.evaluable == 1]
    fila = {"condicion": cond, "nivel": d, "n_sensores": len(ev), "ap_medio": ev.pr_auc.mean()}
    for t in ("puntual", "contextual", "colectiva"):
        fila[f"ap_{t}"] = df[f"pr_auc_{t}"].mean()
    return fila


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niveles", nargs="+", type=float, default=list(NIVELES))
    ap.add_argument("--redes", nargs="*", default=[])
    ap.add_argument("--etiqueta", default="base")
    ap.add_argument("--humo", action="store_true")
    a = ap.parse_args()
    from datos import Fuente, _stats
    import semana10_prueba_final as P10
    import semana11_controles as C
    from g0_linea_base import scores_g0

    f = Fuente("B", "final")
    assert f.rangos["train"][1] == 13593
    # ---- entrenar LR/GBM con el entrenamiento estandar (antes de modificar la prueba)
    modelos = {}
    paso = 200 if a.humo else 6
    t, s, _ = f.indices("train"); k = (t % paso) == 0; t, s = t[k], s[k]
    y = f.labels[t, s].astype(int); ok = y != -1; t, s, y = t[ok], s[ok], y[ok]
    for (mod, ent), nombre in C.NOMBRE.items():
        X = C.X_de(f, t, s, ent)
        m = C.construir(mod); m.fit(X, y); modelos[nombre] = (m, ent)
        print("entrenado", nombre, flush=True)

    base = np.load(P / "pems04_semana2.npz")["data"].astype(np.float64)       # original (T,S,3)
    alt = np.load(P / "esquema_B_final.npz")["data"].astype(np.float64)       # Esquema B
    st = _stats("final")
    a0, b0 = f.rangos["test"]
    L = f.labels[a0:b0]; T = P10.tipos_b()[a0:b0]
    wg0 = int(P10._j("configs/config_g0.json")["seleccion_observada"]["esquema_B"]["ventana"])
    X3_orig = f.X3.copy()
    filas = []
    niveles = [0.5] if a.humo else a.niveles
    for d in niveles:
        mezcla = base + d * (alt - base)                                       # (T,S,3) en unidades originales
        f.X3 = X3_orig.copy()
        f.X3[a0:b0] = ((mezcla[a0:b0] - st[:, 0]) / st[:, 1]).astype(np.float32)
        # G0
        Sg = scores_g0(mezcla[:, :, 0], wg0)[a0:b0]
        filas.append(resumen("G0", d, Sg, L, T, P10)); print(filas[-1], flush=True)
        # LR / GBM
        tt, ss, _ = f.indices("test")
        if a.humo:
            kk = (tt % 50) == 0; tt, ss = tt[kk], ss[kk]
        for nombre, (m, ent) in modelos.items():
            p = m.predict_proba(C.X_de(f, tt, ss, ent))[:, 1].astype(np.float32)
            M = np.full((b0 - a0, f.S), np.nan, dtype=np.float32); M[tt - a0, ss] = p
            filas.append(resumen(nombre, d, M, L, T, P10)); print(filas[-1], flush=True)
        # redes selladas
        for arq in a.redes:
            t0 = time.perf_counter()
            M, L2 = P10.puntuar_modelo(arq, "B", 42, "test", f)
            assert np.array_equal(L2, L)
            filas.append(resumen(COND_RED[arq], d, M, L, T, P10)); print(filas[-1], f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(filas).to_csv(OUT / f"ap_por_nivel_{'HUMO' if a.humo else a.etiqueta}.csv", index=False)
    print(pd.DataFrame(filas).pivot(index="condicion", columns="nivel", values="ap_medio").round(3))


if __name__ == "__main__":
    main()
