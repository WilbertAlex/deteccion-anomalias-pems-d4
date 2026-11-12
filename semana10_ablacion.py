"""
Semana 10 - Ablacion de canales del Esquema B (PPI 2.7.1 y Decision 15, punto 2).

Para cada arquitectura SUPERVISADA (G1 LSTM, G2 GRU, G3 CNN-1D, G5 Transformer, G6 CNN-LSTM-GRU; el autoencoder
no usa etiquetas y queda fuera) repite la corrida principal del Esquema B con solo flujo y variables temporales:
ocupacion y velocidad quedan ANULADAS (valor 0 despues del z-score, es decir, la media del entrenamiento).
  - misma configuracion ganadora de la busqueda y mismas epocas que el reentrenamiento final;
  - todas las ventanas (paso_t = 1), 1 semilla (42), sin parada temprana, sobre el bloque de 80 % (0-13593);
  - se evalua sobre la prueba 13593-16992 (ya abierta; este script exige results/prueba/LOCK.json) con PR-AUC por
    bloque sensor, y tambien por tipo de anomalia (puntual, contextual, colectiva).
Es la comparacion de entrada mas cercana a G0, que solo ve el flujo. La regla esta fijada antes de ver los
resultados y no se modifica en funcion de ellos (Decision 15, punto 5).

Uso (desde la raiz del proyecto, con el venv activo):
  python semana10_ablacion.py --plan                       # configuracion, epocas y horas esperadas por arquitectura
  python semana10_ablacion.py --humo --arq gru             # prueba de humo: 1 epoca, pocas ventanas, se borra
  python semana10_ablacion.py --arq gru                    # una arquitectura (repartir entre laptops)
  python semana10_ablacion.py --compilar                   # une lo hecho; escribe el CSV para la estadistica
Es reanudable: una arquitectura terminada no se repite (results/ablacion/parciales/<arq>_B_s42.csv).
Despues de --compilar:
  python semana9_estadistica.py --etiqueta ablacion --entrada results/ablacion/por_bloque_con_G0.csv --referencia G0 --conjunto prueba
Para repartir entre laptops: cada una corre --arq con una arquitectura distinta y copia a results/ablacion/ los
archivos parciales/<arq>_B_s42.csv, <arq>_B.jsonl y modelos/ a la laptop donde se compila.
"""
import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

ARQS = ("lstm", "gru", "cnn1d", "transformer", "cnn_lstm_gru")          # supervisadas
COND = {"lstm": "G1", "gru": "G2", "cnn1d": "G3", "transformer": "G5", "cnn_lstm_gru": "G6"}
SEMILLA = 42
OUT = Path("results/ablacion")
FIN = Path("results/final")
PRUEBA = Path("results/prueba")


def _j(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def _fuente_ablada():
    """Fuente del Esquema B (bloque final) con ocupacion y velocidad anuladas."""
    from datos import Fuente
    f = Fuente("B", "final")
    assert f.rangos["train"][1] == 13593, f"el bloque 80 % deberia terminar en 13593, termina en {f.rangos['train'][1]}"
    f.X3[:, :, 1:] = 0.0                                              # canales 1 y 2 = ocupacion y velocidad
    t, s, _ = f.indices("train", 200)
    w = f.ventanas_numpy(t[:50], s[:50])
    assert w.shape[2] == 8 and np.all(w[:, :, 1:3] == 0.0), "la anulacion de canales no quedo en las ventanas"
    assert np.any(w[:, :, 0] != 0.0) and np.any(w[:, :, 3:] != 0.0), "el flujo y las variables temporales deben conservarse"
    return f


def horas_esperadas(arq):
    """Horas del reentrenamiento final de la misma arquitectura (Esquema B, mediana de semillas)."""
    r = FIN / f"{arq}_B.jsonl"
    if not r.exists():
        return None
    seg = [json.loads(l)["segundos"] for l in r.read_text(encoding="utf-8").splitlines() if l.strip()]
    return float(np.median(seg)) / 3600 if seg else None


def plan():
    from semana10_reentrenamiento import ganadora
    print(f"{'arq':13s} {'cfg':>3s} {'epocas':>6s} {'h esperadas':>11s}  hecha")
    for arq in ARQS:
        g = ganadora(arq, "B")
        if g is None or g.get("incompleta"):
            print(f"{arq:13s} busqueda incompleta"); continue
        h = horas_esperadas(arq)
        hecha = (OUT / "parciales" / f"{arq}_B_s{SEMILLA}.csv").exists()
        print(f"{arq:13s} {g['cfg_id']:3d} {g['epocas']:6d} {('%.1f' % h) if h else '?':>11s}  {'si' if hecha else 'no'}")
    print("\nLas horas esperadas son las del reentrenamiento final de esa arquitectura en el Esquema B (1 semilla).")


def correr(arq, humo=False):
    import tensorflow as tf
    from datos import dataset
    from entrenar import fijar_semillas, dispositivos
    from modelos import construir
    from semana7_busqueda import puntuar
    from semana10_reentrenamiento import ganadora
    import semana10_prueba_final as P10

    if not (PRUEBA / "LOCK.json").exists():
        sys.exit("Falta results/prueba/LOCK.json: la ablacion se evalua sobre la prueba, que debe estar abierta y cerrada antes.")
    bus = _j("configs/config_busqueda.json")
    g = ganadora(arq, "B")
    if g is None or g.get("incompleta"):
        sys.exit(f"{arq}|B: la busqueda no esta completa")
    destino = Path("results/ablacion_HUMO") if humo else OUT
    if humo:
        shutil.rmtree(destino, ignore_errors=True)
    (destino / "parciales").mkdir(parents=True, exist_ok=True)
    (destino / "modelos").mkdir(parents=True, exist_ok=True)
    ruta_csv = destino / "parciales" / f"{arq}_B_s{SEMILLA}.csv"
    if ruta_csv.exists():
        print(f"{arq}: ya estaba ({ruta_csv.name}), se salta"); return
    epocas, paso_t = (1, 200) if humo else (g["epocas"], 1)
    print(f"{arq}|B ablacion: cfg {g['cfg_id']} {g['cfg']} -> {epocas} epocas, paso_t {paso_t}, semilla {SEMILLA}", flush=True)

    fijar_semillas(SEMILLA)
    f = _fuente_ablada()
    dtr, ntr = dataset(f, "train", batch=bus["lote"], semilla=SEMILLA, paso_t=paso_t)
    m = construir(arq, g["cfg"])
    t0 = time.perf_counter()
    h = m.fit(dtr, epochs=epocas, verbose=2)
    seg = time.perf_counter() - t0
    perdida = [float(v) for v in h.history["loss"]]
    assert all(np.isfinite(perdida)), f"perdida no finita: {perdida}"
    m.save_weights(str(destino / "modelos" / f"{arq}_B_s{SEMILLA}_abl.weights.h5"))

    M, a, b = puntuar(m, arq, f, "test", paso_t=50 if humo else 1)
    L = f.labels[a:b]
    T = P10.tipos_b()[a:b]
    if not np.array_equal(T > 0, L == 1):
        sys.exit("`tipos` no coincide con las etiquetas del Esquema B en la prueba")
    df = P10.metricas_bloques(M, L, np.inf, np.inf, T)
    cols = ["sensor_idx", "n_validos", "n_positivos", "evaluable", "pr_auc", "roc_auc"] + P10.METRICAS_TIPO + P10.NPOS_TIPO
    df = df[cols]
    df.insert(0, "semilla", SEMILLA)
    df.insert(0, "esquema", "B")
    df.insert(0, "condicion", COND[arq] + "-abl")
    df.to_csv(ruta_csv, index=False)
    with open(destino / f"{arq}_B.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"arq": arq, "esquema": "B", "semilla": SEMILLA, "ablacion": "solo flujo + temporales",
                             "cfg_id": g["cfg_id"], "cfg": g["cfg"], "epocas": epocas, "paso_t": paso_t,
                             "ventanas_train": int(ntr), "segundos": seg, "perdida_por_epoca": perdida,
                             "maquina": platform.node(), "procesador": platform.processor(), **dispositivos()},
                            ensure_ascii=False) + "\n")
    ev = df[df.evaluable == 1]
    print(f"ABLACION {arq} B: {epocas} ep, {seg / 3600:.2f} h, PR-AUC medio por bloque {ev.pr_auc.mean():.4f} "
          f"({len(ev)} bloques evaluables)", flush=True)
    tf.keras.backend.clear_session()
    if humo:
        assert len(ev) > 0 and np.isfinite(ev.pr_auc).all()
        shutil.rmtree(destino)
        print("Prueba de humo de la ablacion: OK (se borro lo generado)")


def compilar():
    ps = sorted((OUT / "parciales").glob("*_B_s*.csv"))
    if not ps:
        sys.exit("Aun no hay ablaciones terminadas en results/ablacion/parciales/")
    d = pd.concat([pd.read_csv(p) for p in ps], ignore_index=True)
    d.to_csv(OUT / "por_bloque_metricas.csv", index=False)
    stats = d[["condicion", "esquema"]].copy()
    stats["split"] = "prueba"
    for c in ("sensor_idx", "pr_auc", "n_validos", "n_positivos", "evaluable"):
        stats[c] = d[c]
    stats.to_csv(OUT / "por_bloque.csv", index=False)
    g0 = pd.read_csv(PRUEBA / "por_bloque.csv")
    g0 = g0[(g0.condicion == "G0") & (g0.esquema == "B")]
    pd.concat([g0, stats], ignore_index=True).to_csv(OUT / "por_bloque_con_G0.csv", index=False)
    pm = pd.read_csv(PRUEBA / "por_bloque_metricas.csv")
    print(f"{'condicion':10s} {'PR-AUC abl.':>11s} {'PR-AUC completa':>15s} {'dif.':>8s}  {'puntual':>8s} {'contextual':>10s} {'colectiva':>9s}")
    for cond, x in d.groupby("condicion"):
        ev = x[x.evaluable == 1]
        base = cond[:-4]
        pr_full = pm[(pm.condicion == base) & (pm.esquema == "B") & (pm.evaluable == 1)].pr_auc.mean()
        tm = [ev[c].mean() for c in ("pr_auc_puntual", "pr_auc_contextual", "pr_auc_colectiva")]
        print(f"{cond:10s} {ev.pr_auc.mean():11.4f} {pr_full:15.4f} {ev.pr_auc.mean() - pr_full:+8.4f}  {tm[0]:8.4f} {tm[1]:10.4f} {tm[2]:9.4f}")
    g0m = g0[g0.evaluable == 1].pr_auc.mean()
    print(f"G0 (solo flujo): PR-AUC medio por bloque {g0m:.4f}")
    print(f"\nGuardado en {OUT}: por_bloque_metricas.csv, por_bloque.csv, por_bloque_con_G0.csv")
    print("Siguiente: python semana9_estadistica.py --etiqueta ablacion --entrada results/ablacion/por_bloque_con_G0.csv "
          "--referencia G0 --conjunto prueba")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arq", choices=ARQS)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--humo", action="store_true")
    ap.add_argument("--compilar", action="store_true")
    a = ap.parse_args()
    if a.plan:
        return plan()
    if a.compilar:
        return compilar()
    if not a.arq:
        sys.exit("indica --arq (o usa --plan / --compilar)")
    correr(a.arq, a.humo)


if __name__ == "__main__":
    main()
