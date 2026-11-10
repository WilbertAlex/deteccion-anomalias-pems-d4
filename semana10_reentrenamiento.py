"""
Semana 10 - Reentrenamiento final de G1-G6 (PPI 2.7.1, paso 3; configs/config_busqueda.json, Decision 16).

Para cada (arquitectura, esquema) toma la configuracion GANADORA de la busqueda (results/busqueda/<arq>_<esq>.jsonl,
mismo criterio que `semana7_busqueda.py --resumen`) y la reentrena sobre el bloque 80 % inicial completo:
  - todas las ventanas (paso_t = 1), semillas 42, 43 y 44 (config_busqueda.json: semillas_finales);
  - epocas = mediana de `mejor_epoca` de la ganadora en sus 3 pliegues de busqueda; SIN parada temprana;
  - normalizacion con la entrada "final" (pasos 0-13593) y, en el Esquema B, esquema_B_final.npz;
  - G4 (autoencoder) solo con ventanas de etiqueta 0, como en la busqueda.
NO toca la prueba final: guarda los pesos y los scores del ultimo bloque de validacion del pliegue 3
(10194-13593), que es donde se elige el umbral operativo (2.7.1, paso 4). La prueba se abre en otro script.

Uso (desde la raiz del proyecto, con el venv activo):
  python semana10_reentrenamiento.py --plan                              # ganadoras y epocas de las 12 combinaciones
  python semana10_reentrenamiento.py --prueba --arq cnn1d --esquema A    # humo: 1 epoca, 1/200 ventanas, se borra
  python semana10_reentrenamiento.py --arq cnn1d --esquema A             # las 3 semillas
  python semana10_reentrenamiento.py --arq lstm --esquema B --semillas 43   # solo una semilla (repartir entre laptops)
  python semana10_reentrenamiento.py --resumen                           # que corridas finales hay

Es reanudable: cada semilla terminada se agrega a results/final/<arq>_<esq>.jsonl y se salta si ya esta.
Salidas por corrida: results/final/modelos/<arq>_<esq>_s<semilla>.weights.h5
                     results/final/scores/<arq>_<esq>_s<semilla>.npz   (scores del bloque de umbral, T x S)
"""
import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np

BUS = Path("results/busqueda")
SAL = Path("results/final")
ARQS = ("autoencoder", "cnn1d", "cnn_lstm_gru", "gru", "lstm", "transformer")


def _j(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def ganadora(arq, esq):
    """Configuracion ganadora (argmax de la media de PR-AUC sobre 3 pliegues) y epocas del reentrenamiento."""
    ruta = BUS / f"{arq}_{esq}.jsonl"
    if not ruta.exists():
        return None
    filas = {}
    for ln in ruta.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            filas.setdefault(r["cfg_id"], []).append(r)
    completas = {c: rs for c, rs in filas.items() if len({r["pliegue"] for r in rs}) == 3}
    if len(completas) < len(filas) or len(filas) == 0:
        return {"incompleta": True, "configs_completas": len(completas), "configs": len(filas)}
    cid = max(completas, key=lambda c: float(np.mean([r["pr_auc_media_bloques"] for r in completas[c]])))
    rs = sorted(completas[cid], key=lambda r: r["pliegue"])
    mejores = [int(r["mejor_epoca"]) for r in rs]
    return {"incompleta": False, "cfg_id": cid, "cfg": rs[0]["cfg"],
            "pr_auc_busqueda": float(np.mean([r["pr_auc_media_bloques"] for r in rs])),
            "mejor_epoca_por_pliegue": mejores, "epocas": int(statistics.median(mejores))}


def hechos(ruta):
    if not ruta.exists():
        return set()
    return {json.loads(l)["semilla"] for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()}


def entrenar_final(arq, esq, g, semilla, bus, paso_t, epocas, dir_sal):
    import tensorflow as tf
    from datos import Fuente, dataset, rangos
    from entrenar import fijar_semillas, dispositivos
    from modelos import construir
    from semana7_busqueda import puntuar

    fijar_semillas(semilla)
    f = Fuente(esq, "final")
    assert f.rangos["train"][1] == 13593, f"el bloque 80 % deberia terminar en 13593, termina en {f.rangos['train'][1]}"
    f.rangos["umbral"] = tuple(rangos(3)["val"])                    # ultimo bloque de validacion del pliegue 3
    dtr, ntr = dataset(f, "train", batch=bus["lote"], semilla=semilla, paso_t=paso_t)
    m = construir(arq, g["cfg"])
    if arq == "autoencoder":
        dtr = dtr.unbatch().filter(lambda x, y: y == 0).map(lambda x, y: (x, x)).batch(bus["lote"]).prefetch(tf.data.AUTOTUNE)
    t0 = time.perf_counter()
    h = m.fit(dtr, epochs=epocas, verbose=2)
    seg = time.perf_counter() - t0
    perdida = [float(v) for v in h.history["loss"]]
    assert all(np.isfinite(perdida)), f"perdida no finita: {perdida}"

    nombre = f"{arq}_{esq}_s{semilla}"
    (dir_sal / "modelos").mkdir(parents=True, exist_ok=True)
    (dir_sal / "scores").mkdir(parents=True, exist_ok=True)
    m.save_weights(str(dir_sal / "modelos" / f"{nombre}.weights.h5"))
    M, a, b = puntuar(m, arq, f, "umbral", paso_t=1)
    np.savez_compressed(dir_sal / "scores" / f"{nombre}.npz", score=M, inicio=a, fin=b,
                        labels=f.labels[a:b], split="umbral_val_pliegue3")
    return {"arq": arq, "esquema": esq, "semilla": semilla, "cfg_id": g["cfg_id"], "cfg": g["cfg"],
            "epocas": epocas, "mejor_epoca_por_pliegue": g["mejor_epoca_por_pliegue"], "paso_t": paso_t,
            "ventanas_train": int(ntr), "segundos": seg, "perdida_por_epoca": perdida,
            "bloque_entrenamiento": list(f.rangos["train"]), "bloque_umbral": [int(a), int(b)],
            "maquina": platform.node(), "procesador": platform.processor(), **dispositivos()}


def plan(bus):
    print(f"{'combinacion':22s} {'cfg':>3s} {'PR-AUC busq.':>12s}  mejor_epoca (p1,p2,p3) -> epocas   finales hechas")
    for esq in ("A", "B"):
        for arq in ARQS:
            g = ganadora(arq, esq)
            k = f"{arq}|{esq}"
            if g is None:
                print(f"{k:22s} sin busqueda")
            elif g["incompleta"]:
                print(f"{k:22s} busqueda incompleta ({g['configs_completas']}/{g['configs']} configuraciones con 3 pliegues)")
            else:
                n = len(hechos(SAL / f"{arq}_{esq}.jsonl"))
                print(f"{k:22s} {g['cfg_id']:3d} {g['pr_auc_busqueda']:12.4f}  {str(tuple(g['mejor_epoca_por_pliegue'])):22s}"
                      f" -> {g['epocas']:2d}        {n}/{len(bus['semillas_finales'])}")


def resumen():
    filas = []
    for p in sorted(SAL.glob("*.jsonl")):
        filas += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not filas:
        return print("Aun no hay reentrenamientos finales.")
    for r in sorted(filas, key=lambda r: (r["esquema"], r["arq"], r["semilla"])):
        print(f"{r['arq']:13s} {r['esquema']} semilla {r['semilla']}: cfg {r['cfg_id']}, {r['epocas']} ep, "
              f"{r['segundos'] / 3600:.2f} h, {r['ventanas_train']} ventanas, maquina {r['maquina']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arq", choices=ARQS)
    ap.add_argument("--esquema", choices=["A", "B"])
    ap.add_argument("--semillas", type=int, nargs="+", help="por defecto, semillas_finales de config_busqueda.json")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--prueba", action="store_true")
    a = ap.parse_args()
    bus = _j("configs/config_busqueda.json")
    if a.plan:
        return plan(bus)
    if a.resumen:
        return resumen()
    assert a.arq and a.esquema, "indica --arq y --esquema (o usa --plan / --resumen)"
    g = ganadora(a.arq, a.esquema)
    if g is None or g["incompleta"]:
        sys.exit(f"{a.arq}|{a.esquema}: la busqueda no esta completa; no se puede reentrenar todavia.")
    semillas = a.semillas or bus["semillas_finales"]
    assert set(semillas) <= set(bus["semillas_finales"]), f"semillas permitidas: {bus['semillas_finales']}"

    if a.prueba:                               # humo: nada de esto cuenta como resultado
        dir_sal, paso_t, epocas, semillas = SAL.parent / "final_PRUEBA", 200, 1, semillas[:1]
        shutil.rmtree(dir_sal, ignore_errors=True)
    else:
        dir_sal, paso_t, epocas = SAL, 1, g["epocas"]
    dir_sal.mkdir(parents=True, exist_ok=True)
    ruta = dir_sal / f"{a.arq}_{a.esquema}.jsonl"
    ya = hechos(ruta)
    print(f"{a.arq}|{a.esquema}: cfg {g['cfg_id']} {g['cfg']}  mejor_epoca {g['mejor_epoca_por_pliegue']} -> {epocas} epocas,"
          f" paso_t {paso_t}, semillas {semillas}", flush=True)
    for sem in semillas:
        if sem in ya:
            print(f"{a.arq} {a.esquema} semilla {sem}: ya estaba, se salta", flush=True)
            continue
        r = entrenar_final(a.arq, a.esquema, g, sem, bus, paso_t, epocas, dir_sal)
        with open(ruta, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"FINAL {a.arq} {a.esquema} semilla {sem}: {epocas} ep, {r['segundos'] / 3600:.2f} h, "
              f"{r['ventanas_train']} ventanas, perdida final {r['perdida_por_epoca'][-1]:.4f}", flush=True)

    if a.prueba:
        rs = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
        with np.load(dir_sal / "scores" / f"{a.arq}_{a.esquema}_s{semillas[0]}.npz") as z:   # cierra el archivo (Windows)
            sc, y, inicio = z["score"].copy(), z["labels"].copy(), int(z["inicio"])
        ok = (y != -1) & ~np.isnan(sc)
        assert len(rs) == 1 and sc.shape == (3399, y.shape[1]) and inicio == 10194
        assert ok.sum() > 0 and np.isfinite(sc[ok]).all(), "scores no finitos en el bloque de umbral"
        assert (dir_sal / "modelos" / f"{a.arq}_{a.esquema}_s{semillas[0]}.weights.h5").exists()
        shutil.rmtree(dir_sal)
        print(f"Prueba de humo del reentrenamiento final: OK ({a.arq} {a.esquema}; pesos y scores guardados y borrados)")


if __name__ == "__main__":
    main()
