"""
Semana 11 - Controles no profundos (revision del asesor, obs. 1 y 14). ANALISIS POSTERIOR Y EXPLORATORIO:
se decide el 16 nov 2026, despues de abrir la prueba; debe registrarse asi en la bitacora.

Entrena, SOLO en el Esquema B y con el bloque de 80 % (0-13593), dos clasificadores supervisados no profundos sobre la
MISMA ventana 12 x 8 aplanada que ven las redes, y los evalua una vez sobre la prueba con el mismo AP por sensor:
  LR  = regresion logistica (class_weight balanceado)
  GBM = HistGradientBoostingClassifier
en dos variantes de entrada:
  completo  = 8 canales
  soloflujo = flujo + variables temporales (ocupacion y velocidad en cero, igual que la ablacion)
Guarda ademas los PUNTAJES por ventana en prueba (results/controles/puntajes_<cond>.npy) -> obs. 14.

Uso (raiz del proyecto, venv activo):
  python semana11_controles.py --humo                 # 1 min: pocas ventanas, no escribe nada definitivo
  python semana11_controles.py --modelo lr   --entrada soloflujo
  python semana11_controles.py --modelo lr   --entrada completo
  python semana11_controles.py --modelo gbm  --entrada soloflujo
  python semana11_controles.py --modelo gbm  --entrada completo
  python semana11_controles.py --compilar             # une y escribe results/controles/por_bloque_con_G0.csv
Despues:
  python semana9_estadistica.py --etiqueta controles --entrada results/controles/por_bloque_con_G0.csv --referencia G0 --conjunto prueba
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np
import pandas as pd

OUT = Path("results/controles"); PRUEBA = Path("results/prueba")
SEMILLA = 42
NOMBRE = {("lr", "completo"): "C1-LR", ("lr", "soloflujo"): "C2-LR-flujo",
          ("gbm", "completo"): "C3-GBM", ("gbm", "soloflujo"): "C4-GBM-flujo"}


def construir(modelo):
    if modelo == "lr":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", C=1.0))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                          early_stopping=False, random_state=SEMILLA)


def X_de(f, t, s, entrada, lote=200000):
    out = []
    for i in range(0, len(t), lote):
        w = f.ventanas_numpy(t[i:i + lote], s[i:i + lote]).astype(np.float32)   # (n, 12, 8)
        if entrada == "soloflujo":
            w[:, :, 1:3] = 0.0
        out.append(w.reshape(len(w), -1))
    return np.concatenate(out)


def correr(modelo, entrada, humo=False):
    from datos import Fuente
    import semana10_prueba_final as P10
    if not (PRUEBA / "LOCK.json").exists():
        sys.exit("Falta results/prueba/LOCK.json (la prueba debe estar abierta).")
    cond = NOMBRE[(modelo, entrada)]
    destino = OUT / ("HUMO" if humo else ""); destino.mkdir(parents=True, exist_ok=True)
    csv = destino / f"{cond}_B_s{SEMILLA}.csv"
    if csv.exists():
        print(cond, "ya estaba"); return
    f = Fuente("B", "final")
    assert f.rangos["train"][1] == 13593
    paso = 200 if humo else 6                      # 1 de cada 6 instantes de entrenamiento (declararlo en el texto)
    t, s, _ = f.indices("train")
    k = (t % paso) == 0                            # submuestreo por instante (compatible con cualquier version de datos.py)
    t, s = t[k], s[k]
    y = f.labels[t, s].astype(int)
    ok = y != -1
    t, s, y = t[ok], s[ok], y[ok]
    X = X_de(f, t, s, entrada)
    print(f"{cond}: {len(y)} ventanas de entrenamiento, positivos {y.mean():.3%}, {X.shape[1]} atributos", flush=True)
    m = construir(modelo)
    t0 = time.perf_counter(); m.fit(X, y); seg = time.perf_counter() - t0
    a, b = f.rangos["test"]
    tt, ss, _ = f.indices("test")
    if humo:
        k = (tt % 50) == 0; tt, ss = tt[k], ss[k]
    p = m.predict_proba(X_de(f, tt, ss, entrada))[:, 1].astype(np.float32)
    M = np.full((b - a, f.S), np.nan, dtype=np.float32); M[tt - a, ss] = p
    if not humo:
        np.save(destino / f"puntajes_{cond}.npy", M)
    L = f.labels[a:b]; T = P10.tipos_b()[a:b]
    df = P10.metricas_bloques(M, L, np.inf, np.inf, T)
    cols = ["sensor_idx", "n_validos", "n_positivos", "evaluable", "pr_auc", "roc_auc"] + P10.METRICAS_TIPO + P10.NPOS_TIPO
    df = df[cols]; df.insert(0, "semilla", SEMILLA); df.insert(0, "esquema", "B"); df.insert(0, "condicion", cond)
    df.to_csv(csv, index=False)
    ev = df[df.evaluable == 1]
    print(f"{cond}: AP medio por sensor {ev.pr_auc.mean():.3f} (n={len(ev)}), entrenamiento {seg/60:.1f} min", flush=True)
    (destino / f"{cond}.json").write_text(json.dumps({"cond": cond, "paso_train": paso, "ventanas": int(len(y)),
                                                     "segundos": seg, "semilla": SEMILLA}, indent=1), encoding="utf-8")


def compilar():
    partes = [pd.read_csv(p) for p in sorted(OUT.glob("C*_B_s42.csv"))]
    if not partes: sys.exit("No hay parciales")
    g0 = pd.read_csv("results/ablacion/por_bloque_con_G0.csv")
    g0 = g0[g0.condicion == "G0"]
    res = pd.concat([g0] + partes, ignore_index=True)
    res.to_csv(OUT / "por_bloque_con_G0.csv", index=False)
    print(res.groupby("condicion").apply(lambda d: d[d.evaluable == 1].pr_auc.mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["lr", "gbm"]); ap.add_argument("--entrada", choices=["completo", "soloflujo"])
    ap.add_argument("--humo", action="store_true"); ap.add_argument("--compilar", action="store_true")
    a = ap.parse_args()
    if a.compilar: compilar()
    elif a.humo: correr("lr", "soloflujo", True); correr("gbm", "completo", True)
    else:
        if not (a.modelo and a.entrada): sys.exit("Indica --modelo y --entrada")
        correr(a.modelo, a.entrada)
