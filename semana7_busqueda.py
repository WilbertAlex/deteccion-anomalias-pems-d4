"""
Semana 7 - Busqueda aleatoria de hiperparametros para G1-G6 con la reduccion declarada en
configs/config_busqueda.json (propuesta: 1 semilla, <=10 epocas con parada temprana, una de cada 4 ventanas).

Uso:
  python semana7_busqueda.py --arq lstm --esquema B            # busqueda real (requiere Cuadro 6 en config_modelos.json)
  python semana7_busqueda.py --resumen                         # tabla de ganadores a partir de results/busqueda/*.jsonl
  python semana7_busqueda.py --prueba                          # humo: 2 configuraciones diminutas, 1 pliegue, sin guardar ganadores

Es reanudable: cada entrenamiento terminado se agrega a results/busqueda/<arq>_<esquema>.jsonl y se salta si ya esta.
Validacion: SOLO el bloque de validacion de cada pliegue; la prueba final no se toca aqui.
G4 (autoencoder) se entrena solo con ventanas de etiqueta 0 y se puntua por error de reconstruccion.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np

SAL = Path("results/busqueda")


def _j(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def muestrear(espacio, n, semilla, inicial=None):
    """n configuraciones distintas y deterministas. Si se da `inicial` (Cuadro 6), es la configuracion 0 y
    las n-1 restantes se muestrean del espacio {clave: [valores]} sin repetirla."""
    claves = sorted(espacio)
    for k in claves:
        assert isinstance(espacio[k], list) and len(espacio[k]) > 0, f"{k}: lista de candidatos vacia o null"
    todas = [dict(zip(claves, v)) for v in itertools.product(*[espacio[k] for k in claves])]
    if inicial is not None:
        todas = [c for c in todas if c != {k: inicial[k] for k in claves}]
    rng = np.random.default_rng(semilla)
    idx = rng.permutation(len(todas))[:min(n - (1 if inicial is not None else 0), len(todas))]
    elegidas = [todas[i] for i in sorted(idx)]
    return ([dict(inicial)] if inicial is not None else []) + elegidas


def puntuar(modelo, arq, fuente, split, paso_t=1, lote=1024):
    """Matriz de scores (T_rango, S) del split (NaN donde no hay ventana)."""
    import tensorflow as tf
    from datos import dataset
    from modelos import score_ae
    a, b = fuente.rangos[split]
    t, s, _ = fuente.indices(split, paso_t)
    ds, n = dataset(fuente, split, batch=lote, barajar=False, paso_t=paso_t)
    out = []
    for x, _y in ds:
        out.append(score_ae(modelo, x).numpy() if arq == "autoencoder" else modelo(x, training=False).numpy().reshape(-1))
    p = np.concatenate(out)
    assert len(p) == n
    M = np.full((b - a, fuente.S), np.nan, dtype=np.float32)
    M[t - a, s] = p
    return M, a, b


def pr_auc_por_bloque(M, a, b, fuente):
    from sklearn.metrics import average_precision_score
    vals = []
    for s in range(fuente.S):
        y, sc = fuente.labels[a:b, s], M[:, s]
        ok = (y != -1) & ~np.isnan(sc)
        y, sc = y[ok], sc[ok]
        if len(y) and y.min() != y.max():
            vals.append(float(average_precision_score(y, sc)))
    return vals


def entrenar_uno(arq, cfg, esquema, pliegue, semilla, bus, paso_val):
    import tensorflow as tf
    from datos import Fuente, dataset
    from entrenar import fijar_semillas
    from modelos import construir
    fijar_semillas(semilla)
    f = Fuente(esquema, pliegue)
    dtr, ntr = dataset(f, "train", batch=bus["lote"], semilla=semilla, paso_t=bus["paso_t_busqueda"])
    dva, nva = dataset(f, "val", batch=bus["lote"], barajar=False, paso_t=max(paso_val, bus["paso_t_busqueda"]))
    m = construir(arq, cfg)
    if arq == "autoencoder":
        dtr = dtr.unbatch().filter(lambda x, y: y == 0).map(lambda x, y: (x, x)).batch(bus["lote"]).prefetch(tf.data.AUTOTUNE)
        dva = dva.unbatch().filter(lambda x, y: y == 0).map(lambda x, y: (x, x)).batch(bus["lote"]).prefetch(tf.data.AUTOTUNE)
        mon, modo = "val_loss", "min"
    else:
        mon, modo = "val_pr_auc", "max"
    cb = [tf.keras.callbacks.EarlyStopping(monitor=mon, mode=modo, patience=bus["paciencia"], restore_best_weights=True)]
    t0 = time.perf_counter()
    h = m.fit(dtr, validation_data=dva, epochs=bus["epocas_max_busqueda"], callbacks=cb, verbose=0)
    seg = time.perf_counter() - t0
    M, a, b = puntuar(m, arq, f, "val", paso_t=paso_val)
    v = pr_auc_por_bloque(M, a, b, f)
    return {"segundos": seg, "epocas": len(h.history["loss"]), "ventanas_train": int(ntr),
            "pr_auc_media_bloques": float(np.mean(v)) if v else float("nan"), "n_bloques_evaluables": len(v)}


def resumen():
    filas = {}
    for p in sorted(SAL.glob("*.jsonl")):
        for ln in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            filas.setdefault((r["arq"], r["esquema"], r["cfg_id"]), []).append(r)
    ganadores = {}
    for (arq, esq, cid), rs in filas.items():
        if len({r["pliegue"] for r in rs}) < 3:
            continue            # solo configuraciones completas (3 pliegues)
        sc = float(np.mean([r["pr_auc_media_bloques"] for r in rs]))
        if (arq, esq) not in ganadores or sc > ganadores[(arq, esq)]["score"]:
            ganadores[(arq, esq)] = {"cfg_id": cid, "cfg": rs[0]["cfg"], "score": sc}
    out = {f"{a}|{e}": v for (a, e), v in sorted(ganadores.items())}
    Path("results/busqueda/ganadores.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in out.items():
        print(f"{k:22s} cfg {v['cfg_id']:2d}  PR-AUC medio {v['score']:.4f}  {v['cfg']}")
    if not out:
        print("Aun no hay configuraciones completas.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arq")
    ap.add_argument("--esquema", choices=["A", "B"])
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--prueba", action="store_true")
    a = ap.parse_args()
    SAL.mkdir(parents=True, exist_ok=True)
    if a.resumen:
        return resumen()
    bus = _j("configs/config_busqueda.json")
    if a.prueba:                                  # humo: nada de esto es del Cuadro 6
        bus = dict(bus, epocas_max_busqueda=1, paso_t_busqueda=40)
        esp = {"unidades": [[4], [8]], "dropout": [0.0], "lr": [1e-3]}
        cfgs, arq, esq, pliegues, paso_val, ruta = muestrear(esp, 2, 1), "lstm", "B", [1], 40, SAL / "PRUEBA.jsonl"
        ruta.unlink(missing_ok=True)
    else:
        assert a.arq and a.esquema, "indica --arq y --esquema"
        cm = _j("configs/config_modelos.json")
        cfgs = muestrear(cm["espacio"][a.arq], bus["n_configuraciones"],
                         bus["semilla_muestreo"] + sorted(cm["espacio"]).index(a.arq), inicial=cm["inicial"][a.arq])
        arq, esq, pliegues, paso_val, ruta = a.arq, a.esquema, [1, 2, 3], 1, SAL / f"{a.arq}_{a.esquema}.jsonl"
    hechos = set()
    if ruta.exists():
        for ln in ruta.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            hechos.add((r["cfg_id"], r["pliegue"], r["semilla"]))
    for cid, cfg in enumerate(cfgs):
        for pl in pliegues:
            for sem in bus["semillas_busqueda"]:
                if (cid, pl, sem) in hechos:
                    continue
                r = entrenar_uno(arq, cfg, esq, pl, sem, bus, paso_val)
                r.update(arq=arq, esquema=esq, cfg_id=cid, cfg=cfg, pliegue=pl, semilla=sem)
                with open(ruta, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"{arq} {esq} cfg {cid} pliegue {pl} semilla {sem}: {r['epocas']} ep, {r['segundos']:.0f} s, "
                      f"PR-AUC medio por bloque {r['pr_auc_media_bloques']:.4f} ({r['n_bloques_evaluables']} bloques)")
    if a.prueba:
        rs = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines()]
        assert len(rs) == 2 and all(np.isfinite(r["pr_auc_media_bloques"]) and r["n_bloques_evaluables"] > 0 for r in rs)
        print("Prueba de humo de la busqueda: OK (2 entrenamientos, PR-AUC finito)")
        ruta.unlink()


if __name__ == "__main__":
    main()
