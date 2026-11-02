"""
Semana 9 - Script de latencia de inferencia (VD5, H2; PPI sec. 2.8).

Mide, con time.perf_counter, la latencia de inferencia por ventana de G0 y de las seis arquitecturas
(G1-G6) segun el protocolo: calentamiento >= 100 inferencias, >= 1 000 inferencias consecutivas,
escenarios batch = 1 y batch = 291 / 307; mediana, percentil 95 e IC 95 % (bootstrap de la mediana).

QUE MIDE: solo el tiempo del modelo sobre un lote YA cargado en memoria (sin carga de archivos ni
transferencia). Los pesos son los de la inicializacion: la latencia no depende de los valores de los
pesos, solo de la arquitectura y sus hiperparametros. CUANDO se midan las configuraciones ganadoras de la
busqueda, se pasa --config con el JSON de hiperparametros ganadores (mismo formato que "inicial").

DOS MODOS DE LLAMADA (el protocolo no fija cual es el principal; se reportan los dos y la eleccion es una
decision del equipo, antes de ver resultados de las redes):
  - eager      : modelo(x, training=False), como en score_ae de src/modelos.py.
  - tf_function: el mismo modelo envuelto en tf.function (grafo compilado).
G0 se mide con NumPy: el score de un paso t es |x_t - media| / desv. de los w pasos anteriores,
con w = 22 (ventana elegida), para `batch` sensores a la vez. NO es comparable uno a uno con las
redes sin declararlo: su entrada son 23 valores de flujo por sensor, no una ventana (12, 8).
G7 (SI-DMSeq, PyTorch) NO se mide aqui: se mide con su propia implementacion en su entorno aislado.

NO modifica ningun archivo del protocolo. NO sobrescribe: usa una --etiqueta distinta en cada corrida.

Uso (desde la raiz del proyecto):
  python semana9_latencia.py --etiqueta cpu-windows-prueba --medidas 200       # prueba rapida
  python semana9_latencia.py --etiqueta cpu-windows-inicial                    # medicion completa
Genera: results/eda/latencia_<etiqueta>.json
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

ARQ = ["lstm", "gru", "cnn1d", "autoencoder", "transformer", "cnn_lstm_gru"]
SIGMA_MIN = 1e-6  # igual que src/g0_linea_base.py


def resumen(ms, rng, n_boot=2000):
    """Mediana, p95, media e IC 95 % de la mediana (bootstrap percentil) de un vector de ms."""
    ms = np.asarray(ms, dtype=float)
    idx = rng.integers(0, len(ms), size=(n_boot, len(ms)))
    med_boot = np.median(ms[idx], axis=1)
    return {"n": int(len(ms)), "mediana_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "media_ms": float(ms.mean()), "min_ms": float(ms.min()), "max_ms": float(ms.max()),
            "ic95_mediana_ms": [float(np.percentile(med_boot, 2.5)), float(np.percentile(med_boot, 97.5))]}


def medir(f, calentamiento, medidas):
    """Llama f() `calentamiento` veces sin registrar y luego `medidas` veces registrando ms."""
    for _ in range(calentamiento):
        f()
    out = np.empty(medidas)
    for i in range(medidas):
        t0 = time.perf_counter()
        f()
        out[i] = (time.perf_counter() - t0) * 1000.0
    return out


def hardware():
    h = {"sistema": platform.platform(), "procesador": platform.processor(), "python": platform.python_version(),
         "nucleos_logicos": os.cpu_count()}
    try:
        import psutil
        h["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
        h["nucleos_fisicos"] = psutil.cpu_count(logical=False)
    except Exception:
        h["ram_gb"] = None
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiqueta", required=True, help="nombre unico de la corrida, p. ej. cpu-windows-inicial")
    ap.add_argument("--arq", default=",".join(ARQ), help="arquitecturas a medir, separadas por coma")
    ap.add_argument("--no-g0", action="store_true", help="omite G0")
    ap.add_argument("--batches", default="1,291,307")
    ap.add_argument("--calentamiento", type=int, default=100)
    ap.add_argument("--medidas", type=int, default=1000)
    ap.add_argument("--ventana-g0", type=int, default=22)
    ap.add_argument("--config", default=None,
                    help="JSON con hiperparametros por arquitectura (por defecto 'inicial' de configs/config_modelos.json)")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()
    if "prueba" not in a.etiqueta and (a.calentamiento < 100 or a.medidas < 1000):
        sys.exit("El protocolo exige >= 100 de calentamiento y >= 1 000 medidas. "
                 "Para una prueba corta usa una --etiqueta que contenga 'prueba'.")
    out = Path(f"results/eda/latencia_{a.etiqueta}.json")
    if out.exists():
        sys.exit(f"Ya existe {out}. Usa otra --etiqueta para no sobrescribirlo.")
    batches = [int(b) for b in a.batches.split(",")]
    arqs = [x for x in a.arq.split(",") if x]
    rng = np.random.default_rng(a.semilla)

    res = {}
    if not a.no_g0:
        w = a.ventana_g0
        res["g0"] = {}
        for b in batches:
            x = rng.normal(100.0, 20.0, size=(b, w + 1)).astype(np.float64)

            def g0(x=x):
                ref = x[:, :w]
                sd = ref.std(axis=1)
                z = np.abs(x[:, w] - ref.mean(axis=1)) / np.where(sd < SIGMA_MIN, 1.0, sd)
                return np.where(sd < SIGMA_MIN, 0.0, z)
            ms = medir(g0, a.calentamiento, a.medidas)
            res["g0"][f"batch{b}"] = {"numpy": resumen(ms, rng)}
            print(f"g0           batch {b:4d}: mediana {np.median(ms):8.4f} ms  p95 {np.percentile(ms, 95):8.4f} ms")

    meta_tf = None
    if arqs:
        import tensorflow as tf
        import keras
        from modelos import construir, score_ae
        meta_tf = {"tensorflow": tf.__version__, "keras": keras.__version__,
                   "hilos_inter_op": tf.config.threading.get_inter_op_parallelism_threads(),
                   "hilos_intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
                   "dispositivos": [d.name for d in tf.config.list_physical_devices()]}
        src_cfg = Path(a.config) if a.config else Path("configs/config_modelos.json")
        cfgs = json.loads(src_cfg.read_text(encoding="utf-8-sig"))
        cfgs = cfgs if a.config else cfgs["inicial"]
        for arq in arqs:
            tf.keras.utils.set_random_seed(a.semilla)
            m = construir(arq, cfgs[arq])
            res[arq] = {"parametros": int(m.count_params()), "config": cfgs[arq]}

            def llamar(x, m=m, arq=arq):
                return score_ae(m, x) if arq == "autoencoder" else m(x, training=False)
            for b in batches:
                xb = tf.constant(rng.normal(0.0, 1.0, size=(b, 12, 8)).astype(np.float32))
                modos = {"eager": lambda xb=xb: llamar(xb).numpy(),
                         "tf_function": (lambda g=tf.function(llamar), xb=xb: g(xb).numpy())}
                res[arq][f"batch{b}"] = {}
                for nombre, f in modos.items():
                    ms = medir(f, a.calentamiento, a.medidas)
                    res[arq][f"batch{b}"][nombre] = resumen(ms, rng)
                    print(f"{arq:12s} batch {b:4d} {nombre:11s}: mediana {np.median(ms):8.3f} ms  "
                          f"p95 {np.percentile(ms, 95):8.3f} ms")

    # criterios del protocolo (sec. 2.8): <= 10 ms con batch 1; < 5 s con batch 307. Solo se marcan, no se decide.
    criterios = {}
    for k, v in res.items():
        crit = {}
        for modo in ("numpy", "eager", "tf_function"):
            b1 = v.get("batch1", {}).get(modo)
            b307 = v.get("batch307", {}).get(modo)
            if b1:
                crit[f"batch1_{modo}_mediana_<=10ms"] = bool(b1["mediana_ms"] <= 10.0)
            if b307:
                crit[f"batch307_{modo}_mediana_<5000ms"] = bool(b307["mediana_ms"] < 5000.0)
        criterios[k] = crit
    meta = {"etiqueta": a.etiqueta, "hardware": hardware(), "tf": meta_tf, "calentamiento": a.calentamiento,
            "medidas": a.medidas, "batches": batches, "semilla": a.semilla,
            "nota": "Pesos de inicializacion (la latencia no depende de sus valores). Sin carga de archivos. "
                    "G7 no incluido. Modo principal (eager o tf_function) pendiente de decision del equipo."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "resultados": res, "criterios_marcados": criterios},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
