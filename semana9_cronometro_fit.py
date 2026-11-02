"""
Semana 9 - Cronometro con fit(): comprobacion de validez del cronometro de la semana 6.

El cronometro de la semana 6 (semana6_cronometro_epoca.py) mide m.train_on_batch() en un bucle de
Python. La busqueda real (semana7_busqueda.py) entrena con m.fit() sobre tf.data. Los dos caminos
no tienen el mismo costo por paso, asi que la extrapolacion de horas puede no representar la busqueda.

Con el MISMO modelo (configuracion inicial del Cuadro 6, configs/config_modelos.json) y el MISMO
cargador (src/datos.py), este script mide el tiempo por paso en cuatro modos:
  1. solo_datos        : tf.data sin entrenar (costo de producir un lote).
  2. train_on_batch    : bucle de Python, como la semana 6.
  3. fit_spe1          : model.fit(), como la busqueda real.
  4. fit_spe<k>        : model.fit() con steps_per_execution=k (agrupa k pasos en una sola llamada;
                         no cambia hiperparametros ni el numero ni el orden de las actualizaciones).
El tiempo por paso se obtiene como pendiente entre dos longitudes (n1 y n2 pasos): asi se excluyen el
costo fijo de arranque (trazado del grafo, llenado del buffer de barajado).

NO mide: la validacion que fit() hace en cada epoca ni la puntuacion final sobre la validacion completa.
NO modifica ningun archivo del protocolo ni ninguna decision. Si se adopta steps_per_execution en la
busqueda, es un cambio del arnes de entrenamiento y se registra como decision ANTES de ver resultados.

Uso (desde la raiz del proyecto):
  python semana9_cronometro_fit.py --etiqueta cpu-windows-fit
  python semana9_cronometro_fit.py --etiqueta gpu-wsl2-fit --arq transformer --spe 1,8,32,64
Genera: results/eda/cronometro_fit_<etiqueta>.json (no sobrescribe: usa una etiqueta distinta en cada corrida).
"""
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

SUPERVISADAS = ["lstm", "gru", "cnn1d", "autoencoder", "transformer", "cnn_lstm_gru"]  # el autoencoder se entrena solo con y == 0


def compilar(m, arq, cfg, spe):
    """Misma compilacion que modelos._compilar para modelos supervisados, mas steps_per_execution."""
    import tensorflow as tf
    if arq in ("cnn1d", "transformer"):
        opt = tf.keras.optimizers.AdamW(learning_rate=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = tf.keras.optimizers.Adam(cfg["lr"])
    if arq == "autoencoder":
        m.compile(optimizer=opt, loss="mse", steps_per_execution=spe)
    else:
        m.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")], steps_per_execution=spe)
    return m


def pendiente(f, n1, n2):
    """Tiempo por paso (s) como pendiente entre n1 y n2 pasos, y costo fijo (s)."""
    t = {}
    for n in (n1, n2):
        t0 = time.perf_counter()
        f(n)
        t[n] = time.perf_counter() - t0
    p = (t[n2] - t[n1]) / (n2 - n1)
    return p, t[n1] - p * n1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiqueta", required=True, help="nombre unico de la corrida, p. ej. cpu-windows-fit")
    ap.add_argument("--arq", default="lstm", choices=SUPERVISADAS)
    ap.add_argument("--esquema", default="B", choices=["A", "B"])
    ap.add_argument("--pliegue", default="3")
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--paso-t", type=int, default=4, help="submuestreo de la busqueda (config_busqueda.json)")
    ap.add_argument("--n1", type=int, default=256)
    ap.add_argument("--n2", type=int, default=1280)
    ap.add_argument("--spe", default="1,8,32", help="valores de steps_per_execution a probar, separados por coma")
    a = ap.parse_args()
    pliegue = a.pliegue if a.pliegue == "final" else int(a.pliegue)
    spes = [int(x) for x in a.spe.split(",")]
    for k in spes:
        assert a.n1 % k == 0 and a.n2 % k == 0, f"n1 y n2 deben ser multiplos de {k}"
    out = Path(f"results/eda/cronometro_fit_{a.etiqueta}.json")
    if out.exists():
        sys.exit(f"Ya existe {out}. Usa otra --etiqueta para no sobrescribirlo.")

    import numpy as np
    import tensorflow as tf
    from datos import Fuente, dataset
    from entrenar import fijar_semillas, dispositivos
    from modelos import construir

    cfg = json.loads(Path("configs/config_modelos.json").read_text(encoding="utf-8"))["inicial"][a.arq]
    disp = dispositivos()
    print("Dispositivos:", disp)
    f = Fuente(a.esquema, pliegue)
    ds, n = dataset(f, "train", batch=a.lote, semilla=42, paso_t=a.paso_t)
    n_completo = len(f.indices("train")[0])
    pasos_busqueda = math.ceil(n / a.lote)
    pasos_completo = math.ceil(n_completo / a.lote)
    print(f"{a.arq}, esquema {a.esquema}, pliegue {pliegue}: {n:,} ventanas con paso_t={a.paso_t} "
          f"({pasos_busqueda:,} pasos/epoca de busqueda); {n_completo:,} ventanas completas "
          f"({pasos_completo:,} pasos/epoca final), lote {a.lote}")

    if a.arq == "autoencoder":  # igual que semana7_busqueda.py: solo ventanas de etiqueta 0, objetivo = la entrada
        ds = ds.unbatch().filter(lambda x, y: y == 0).map(lambda x, y: (x, x)).batch(a.lote).prefetch(tf.data.AUTOTUNE)
        print("Autoencoder: se entrena solo con y == 0; los pasos/epoca de arriba cuentan todas las ventanas "
              "(sobrestiman unos pocos puntos porcentuales).")
    dsr = ds.repeat()  # evita quedarse sin lotes si n1/n2 superan una epoca
    res = {}

    # 1. tf.data solo
    def solo_datos(k):
        for _ in dsr.take(k):
            pass
    solo_datos(a.n1)  # calentamiento
    p, fijo = pendiente(solo_datos, a.n1, a.n2)
    res["solo_datos"] = {"ms_por_paso": p * 1000, "fijo_s": fijo}
    print(f"solo_datos        : {p * 1000:7.2f} ms/paso")

    # 2. train_on_batch (como la semana 6)
    fijar_semillas(42)
    m = compilar(construir(a.arq, cfg), a.arq, cfg, 1)
    it = iter(dsr)
    for _ in range(30):
        x, y = next(it)
        m.train_on_batch(x, y)
    ts = []
    for _ in range(300):
        t1 = time.perf_counter()
        x, y = next(it)
        m.train_on_batch(x, y)
        ts.append(time.perf_counter() - t1)
    p = float(np.mean(ts))
    res["train_on_batch"] = {"ms_por_paso": p * 1000, "ms_mediana": float(np.median(ts)) * 1000}
    print(f"train_on_batch    : {p * 1000:7.2f} ms/paso (mediana {np.median(ts) * 1000:.2f})")

    # 3 y 4. fit() con distintos steps_per_execution
    for k in spes:
        fijar_semillas(42)
        m = compilar(construir(a.arq, cfg), a.arq, cfg, k)
        m.fit(dsr.take(a.n1), epochs=1, verbose=0)  # calentamiento (traza el grafo)
        p, fijo = pendiente(lambda nn: m.fit(dsr.take(nn), epochs=1, verbose=0), a.n1, a.n2)
        res[f"fit_spe{k}"] = {"steps_per_execution": k, "ms_por_paso": p * 1000, "fijo_s": fijo}
        print(f"fit_spe{k:<10d}: {p * 1000:7.2f} ms/paso (fijo {fijo:.1f} s)")

    base = res["train_on_batch"]["ms_por_paso"]
    for nombre, r in res.items():
        r["epoca_busqueda_min"] = r["ms_por_paso"] * pasos_busqueda / 60000
        r["epoca_final_min"] = r["ms_por_paso"] * pasos_completo / 60000
        r["veces_mas_rapido_que_train_on_batch"] = base / r["ms_por_paso"] if r["ms_por_paso"] > 0 else None

    import keras
    meta = {"etiqueta": a.etiqueta, "dispositivos": disp, "python": platform.python_version(),
            "tensorflow": tf.__version__, "keras": keras.__version__, "arquitectura": a.arq, "cfg": cfg,
            "esquema": a.esquema, "pliegue": str(pliegue), "lote": a.lote, "paso_t": a.paso_t,
            "ventanas_train_busqueda": int(n), "ventanas_train_completas": int(n_completo),
            "pasos_por_epoca_busqueda": pasos_busqueda, "pasos_por_epoca_final": pasos_completo,
            "n1": a.n1, "n2": a.n2,
            "nota": "No incluye validacion por epoca ni puntuacion final. Modelo = configuracion inicial del Cuadro 6."}
    print("\nResumen (minutos por epoca estimados; busqueda = paso_t del config, final = todas las ventanas):")
    print(f"{'modo':18s}{'ms/paso':>9s}{'busqueda':>10s}{'final':>8s}{'x vs t_o_b':>12s}")
    for nombre, r in res.items():
        print(f"{nombre:18s}{r['ms_por_paso']:9.2f}{r['epoca_busqueda_min']:10.2f}{r['epoca_final_min']:8.2f}"
              f"{r['veces_mas_rapido_que_train_on_batch']:12.2f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "resultados": res}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
