"""
Semana 10 - Umbral operativo y prueba final UNICA para G0-G6 (PPI 2.7.1, pasos 4 y 5; Decisiones 16 y 18).
G7 (SI-DMSeq, PyTorch) se evalua con su propio script en .venv-g7 y se une despues (ver "Unir G7").

LO QUE HACE
  Umbral (2.7.1, paso 4): para cada modelo final y semilla, el umbral que maximiza F1 y el que alcanza recall 0.90 se
  eligen SOLO con los scores del bloque 10194-13593 (ultimo bloque de validacion del pliegue 3) que dejo
  semana10_reentrenamiento.py en results/final/scores/. Un umbral global por modelo y semilla (los modelos son
  globales, un solo score por ventana), calculado sobre todos los pares (paso, sensor) validos del bloque, sin usar
  la prueba. G0 usa su k* sellado (Decision 11) para F1 y el umbral de recall 0.90 del bloque.
  Prueba (2.7.1, paso 5): cada modelo final se puntua UNA vez sobre 13593-16992 y se calcula, por bloque sensor,
  PR-AUC, AUC-ROC, F1, FPR y recall (a los dos umbrales). Las tres semillas se resumen con la MEDIA dentro de cada
  bloque (Decision 18, punto 3); se conservan los resultados por semilla.
  Predice "anomalia" cuando score >= umbral.

MODOS (todos desde la raiz del proyecto, con el venv activo)
  python semana10_prueba_final.py --umbrales
        Calcula y guarda results/prueba/umbrales.json. NO toca la prueba.
  python semana10_prueba_final.py --verificar --arq lstm --esquema B [--semilla 42]
        Recarga los pesos, vuelve a puntuar el bloque del umbral y comprueba que coincide con el score guardado
        al reentrenar. Valida la cadena cargar-pesos -> puntuar que usara la prueba. NO toca la prueba.
  python semana10_prueba_final.py --ensayo [--arq lstm] [--esquema B]
        Corre TODO el flujo (incluido el CSV para semana9_estadistica.py) sobre el bloque del umbral en lugar de la
        prueba. Escribe en results/prueba_ENSAYO/ (para probar la estadistica sobre ese CSV usa --etiqueta ensayo).
        Es solo para probar el codigo: ese bloque entro en el
        reentrenamiento, asi que sus numeros NO son resultados y no se reportan.
  python semana10_prueba_final.py --ganadoras
        Escribe results/final/ganadoras_A.json y ganadoras_B.json (hiperparametros finales por arquitectura) para
        pasarlos a semana9_latencia.py --config.
  python semana10_prueba_final.py --prueba --aviso-asesor AAAA-MM-DD
        ABRE LA PRUEBA, UNA SOLA VEZ. Exige la fecha del aviso al asesor, pide escribir una confirmacion, congela
        las huellas (SHA-256) de pesos, configuraciones y umbrales en results/prueba/LOCK.json y se niega a repetirse.
        Si se interrumpe (apagon, memoria), se reanuda con --reanudar: no vuelve a puntuar lo ya hecho y comprueba
        que las huellas no cambiaron.

DESPUES DE LA PRUEBA
  python semana9_estadistica.py --etiqueta prueba --entrada results/prueba/por_bloque.csv --referencia G0 --conjunto prueba
  python semana9_estadistica.py --etiqueta prueba_delta005 --entrada results/prueba/por_bloque.csv --referencia G0 --conjunto prueba --delta 0.05
  (primario sin margen; el margen 0.05 es secundario, Decision 18, punto 2.)
  Unir G7: su script debe escribir un CSV con las mismas columnas que results/prueba/por_bloque.csv (condicion = G7);
  se concatena a ese archivo antes de correr la estadistica.

Salidas en results/prueba/: umbrales.json, parciales/*.csv (un archivo por modelo y semilla), por_bloque_semilla.csv,
por_bloque.csv (formato de semana9_estadistica.py), por_bloque_metricas.csv, metricas.json, LOCK.json.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, "src")

ARQS = ("lstm", "gru", "cnn1d", "autoencoder", "transformer", "cnn_lstm_gru")
COND = {"lstm": "G1", "gru": "G2", "cnn1d": "G3", "autoencoder": "G4", "transformer": "G5", "cnn_lstm_gru": "G6"}
ESQUEMAS = ("A", "B")
SEMILLAS = (42, 43, 44)
FIN = Path("results/final")
OUT = Path("results/prueba")
OUT_ENSAYO = Path("results/prueba_ENSAYO")
P = Path("data/processed")
RECALL_OBJ = 0.90
METRICAS = ["pr_auc", "roc_auc", "f1_f1", "fpr_f1", "recall_f1", "f1_r90", "fpr_r90", "recall_r90"]


def _j(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def _rangos_split():
    part = _j("configs/particion_temporal.json")
    pl3 = next(x for x in part["pliegues"] if x["pliegue"] == 3)
    return {"umbral": tuple(pl3["val_idx"]), "test": tuple(part["test_idx"])}


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# ------------------------------------------------------------------ nucleo numerico (sin TensorFlow)
def umbrales(score, labels):
    """Umbral de F1 maximo y umbral de recall >= RECALL_OBJ sobre un bloque (todos los pares validos juntos).
    Empates: gana el umbral mas bajo en F1; en recall, el mas alto que aun alcanza el recall objetivo."""
    ok = (labels != -1) & ~np.isnan(score)
    y = (labels[ok] == 1).astype(np.int8)
    s = score[ok].astype(np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("el bloque del umbral no tiene ambas clases")
    prec, rec, thr = precision_recall_curve(y, s)
    p, r = prec[:-1], rec[:-1]                                   # p, r alineados con thr
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    i = int(np.argmax(f1))
    j = int(np.nonzero(r >= RECALL_OBJ)[0][-1])
    return {"umbral_f1": float(thr[i]), "f1_en_bloque": float(f1[i]),
            "umbral_recall90": float(thr[j]), "recall_en_bloque": float(r[j]),
            "n_validos": int(len(y)), "n_positivos": int(y.sum())}


def metricas_bloques(score, labels, thr_f1, thr_r90):
    """Una fila por sensor (bloque). score y labels: (n_pasos, S). Un bloque es evaluable si tiene ambas clases."""
    filas = []
    for s in range(score.shape[1]):
        y, sc = labels[:, s], score[:, s]
        ok = (y != -1) & ~np.isnan(sc)
        y, sc = y[ok], sc[ok]
        n, npos = int(len(y)), int((y == 1).sum())
        ev = n > 0 and 0 < npos < n
        fila = {"sensor_idx": s, "n_validos": n, "n_positivos": npos, "evaluable": int(ev)}
        for c in METRICAS:
            fila[c] = np.nan
        if ev:
            fila["pr_auc"] = float(average_precision_score(y, sc))
            fila["roc_auc"] = float(roc_auc_score(y, sc))
            for nombre, t in (("f1", thr_f1), ("r90", thr_r90)):
                pred = sc >= t
                tp = int((pred & (y == 1)).sum())
                fp = int((pred & (y == 0)).sum())
                fn = npos - tp
                tn = (n - npos) - fp
                fila[f"f1_{nombre}"] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
                fila[f"fpr_{nombre}"] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
                fila[f"recall_{nombre}"] = tp / npos
        filas.append(fila)
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ G0
_CACHE_G0 = {}


def g0_scores(esq):
    """Scores de G0 sobre toda la serie (T, S), etiquetas (T, S) y parametros sellados (Decision 11)."""
    if esq not in _CACHE_G0:
        from g0_linea_base import scores_g0
        if esq == "A":
            with np.load(P / "pems04_semana2.npz") as z:
                flujo = z["data"][:, :, 0].astype(np.float64)
            with np.load(P / "labels_esquema_A.npz") as z:
                lab = z["labels"].copy()
        else:
            with np.load(P / "esquema_B_final.npz") as z:
                flujo = z["data"][:, :, 0].astype(np.float64)
                lab = z["labels"].copy()
        p = _j("configs/config_g0.json")["seleccion_observada"][f"esquema_{esq}"]
        _CACHE_G0[esq] = (scores_g0(flujo, int(p["ventana"])), lab, p)
    return _CACHE_G0[esq]


def umbrales_g0(esq):
    S, lab, p = g0_scores(esq)
    a, b = _rangos_split()["umbral"]
    u = umbrales(S[a:b], lab[a:b])
    u["umbral_f1_optimo_del_bloque"] = u["umbral_f1"]          # solo informativo
    u["umbral_f1"] = float(p["multiplicador"])                  # G0 usa su k* sellado (Decision 11)
    u["ventana"] = int(p["ventana"])
    return u


# ------------------------------------------------------------------ modelos finales (TensorFlow)
def cfg_final(arq, esq):
    ruta = FIN / f"{arq}_{esq}.jsonl"
    filas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = {r["cfg_id"] for r in filas}
    sems = sorted({r["semilla"] for r in filas})
    if len(ids) != 1 or sems != list(SEMILLAS):
        raise SystemExit(f"{arq}|{esq}: se esperaba 1 configuracion y semillas {list(SEMILLAS)}; hay cfg {sorted(ids)} y semillas {sems}")
    return filas[0]["cfg"], filas[0]["cfg_id"]


def preparar_fuente(esq):
    from datos import Fuente, rangos
    f = Fuente(esq, "final")
    f.rangos["umbral"] = tuple(rangos(3)["val"])
    return f


def puntuar_modelo(arq, esq, sem, split, fuente):
    """Scores (n_pasos, S) y etiquetas del split ('umbral' o 'test') con los pesos finales de (arq, esq, sem)."""
    import tensorflow as tf
    from modelos import construir
    from semana7_busqueda import puntuar
    cfg, _ = cfg_final(arq, esq)
    m = construir(arq, cfg)
    ruta = FIN / "modelos" / f"{arq}_{esq}_s{sem}.weights.h5"
    if not ruta.exists():
        raise SystemExit(f"falta {ruta}")
    m.load_weights(str(ruta))
    M, a, b = puntuar(m, arq, fuente, split, paso_t=1)
    L = fuente.labels[a:b]
    tf.keras.backend.clear_session()
    return M, L


# ------------------------------------------------------------------ modos
def cmd_umbrales():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    ini = _rangos_split()["umbral"]
    for esq in ESQUEMAS:
        for arq in ARQS:
            cfg_final(arq, esq)
            for s in SEMILLAS:
                ruta = FIN / "scores" / f"{arq}_{esq}_s{s}.npz"
                with np.load(ruta) as z:
                    sc, lab = z["score"].copy(), z["labels"].copy()
                    inicio, fin = int(z["inicio"]), int(z["fin"])
                assert (inicio, fin) == ini, f"{ruta}: bloque {(inicio, fin)} distinto de {ini}"
                u = umbrales(sc, lab)
                u.update(arq=arq, sha256_scores=sha256(ruta))
                res[f"{COND[arq]}|{esq}|{s}"] = u
        res[f"G0|{esq}|0"] = umbrales_g0(esq)
    (OUT / "umbrales.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'modelo|esquema|semilla':22s} {'umbral F1':>12s} {'F1 bloque':>10s} {'umbral R90':>12s} {'recall':>7s}")
    for k, u in res.items():
        print(f"{k:22s} {u['umbral_f1']:12.6g} {u['f1_en_bloque']:10.4f} {u['umbral_recall90']:12.6g} {u['recall_en_bloque']:7.3f}")
    print(f"\nGuardado: {OUT / 'umbrales.json'}  (sin tocar la prueba)")


def cmd_verificar(arq, esq, sem):
    ini = _rangos_split()["umbral"]
    f = preparar_fuente(esq)
    M, L = puntuar_modelo(arq, esq, sem, "umbral", f)
    with np.load(FIN / "scores" / f"{arq}_{esq}_s{sem}.npz") as z:
        G = z["score"].copy()
    assert M.shape == G.shape, f"forma {M.shape} frente a {G.shape}"
    mismos_nan = bool(np.array_equal(np.isnan(M), np.isnan(G)))
    ok = ~np.isnan(M) & ~np.isnan(G)
    dif = float(np.max(np.abs(M[ok] - G[ok]))) if ok.any() else float("nan")
    escala = float(np.max(np.abs(G[ok]))) if ok.any() else float("nan")
    ap_a = metricas_bloques(M, L, 0.5, 0.5)["pr_auc"].mean()
    ap_b = metricas_bloques(G, L, 0.5, 0.5)["pr_auc"].mean()
    bien = mismos_nan and dif <= 1e-4 * max(1.0, escala)
    print(f"{arq}|{esq}|s{sem}: bloque {ini}, {int(ok.sum())} scores; misma mascara de NaN: {mismos_nan}; "
          f"diferencia maxima {dif:.3g} (escala {escala:.3g}); PR-AUC medio por bloque: recargado {ap_a:.6f}, guardado {ap_b:.6f}")
    print("VERIFICACION OK: la carga de pesos y el puntaje reproducen lo guardado." if bien else
          "VERIFICACION FALLIDA: no uses este flujo sobre la prueba hasta entender la diferencia.")
    if not bien:
        sys.exit(1)


def cmd_ganadoras():
    for esq in ESQUEMAS:
        d = {arq: cfg_final(arq, esq)[0] for arq in ARQS}
        ruta = FIN / f"ganadoras_{esq}.json"
        ruta.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Guardado: {ruta}")
    print("Uso: python semana9_latencia.py --etiqueta cpu-windows-final-B --config results/final/ganadoras_B.json")


def evaluar(esq, cond, arq, sem, split, umbr, fuente):
    a, b = _rangos_split()[split]
    if cond == "G0":
        S, lab, _ = g0_scores(esq)
        M, L, clave = S[a:b], lab[a:b], f"G0|{esq}|0"
    else:
        M, L = puntuar_modelo(arq, esq, sem, split, fuente)
        clave = f"{cond}|{esq}|{sem}"
    u = umbr[clave]
    df = metricas_bloques(M, L, u["umbral_f1"], u["umbral_recall90"])
    df.insert(0, "semilla", 0 if cond == "G0" else sem)
    df.insert(0, "esquema", esq)
    df.insert(0, "condicion", cond)
    return df


def compilar(destino, nombre_split, ensayo=False):
    parciales = sorted((destino / "parciales").glob("*.csv"))
    d = pd.concat([pd.read_csv(p) for p in parciales], ignore_index=True)
    d.to_csv(destino / "por_bloque_semilla.csv", index=False)
    agg = {c: "mean" for c in METRICAS}
    agg.update(n_validos="first", n_positivos="first", evaluable="first")
    g = d.groupby(["condicion", "esquema", "sensor_idx"], as_index=False).agg(agg)
    base = g[["condicion", "esquema"]].copy()
    base["split"] = nombre_split
    for c in ("sensor_idx", "pr_auc", "n_validos", "n_positivos", "evaluable"):
        base[c] = g[c]
    base.to_csv(destino / "por_bloque.csv", index=False)
    g.to_csv(destino / "por_bloque_metricas.csv", index=False)
    resumen = {}
    for (cond, esq), x in g.groupby(["condicion", "esquema"]):
        ev = x[x.evaluable == 1]
        por_sem = d[(d.condicion == cond) & (d.esquema == esq) & (d.evaluable == 1)].groupby("semilla").pr_auc.mean()
        resumen[f"{cond}|{esq}"] = {"n_bloques_evaluables": int(len(ev)), "n_bloques": int(len(x)),
                                    "pr_auc_media_por_semilla": {int(k): float(v) for k, v in por_sem.items()},
                                    **{f"{c}_media": float(ev[c].mean()) for c in METRICAS},
                                    "pr_auc_de_media_de_semillas_sd": float(ev.pr_auc.std(ddof=1))}
    salida = {"ensayo": bool(ensayo), "nota": ("ENSAYO sobre el bloque del umbral: NO son resultados" if ensayo else "prueba final"),
              "condiciones": resumen}
    (destino / "metricas.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'condicion|esquema':18s} {'n eval':>6s} {'PR-AUC':>8s} {'ROC':>7s} {'F1@F1':>7s} {'FPR@F1':>8s} {'FPR@R90':>8s}")
    for k, v in resumen.items():
        print(f"{k:18s} {v['n_bloques_evaluables']:6d} {v['pr_auc_media']:8.4f} {v['roc_auc_media']:7.4f} "
              f"{v['f1_f1_media']:7.4f} {v['fpr_f1_media']:8.5f} {v['fpr_r90_media']:8.5f}")
    for esq in ESQUEMAS:
        ns = {v["n_bloques_evaluables"] for k, v in resumen.items() if k.endswith("|" + esq)}
        if len(ns) > 1:
            print(f"AVISO: esquema {esq} tiene distinto numero de bloques evaluables entre condiciones: {sorted(ns)}")
    print(f"\nGuardado en {destino}: por_bloque.csv, por_bloque_semilla.csv, por_bloque_metricas.csv, metricas.json")


def correr(split, destino, arq_f=None, esq_f=None):
    ruta_u = OUT / "umbrales.json"
    if not ruta_u.exists():
        sys.exit("Falta results/prueba/umbrales.json: ejecuta primero --umbrales.")
    umbr = _j(ruta_u)
    (destino / "parciales").mkdir(parents=True, exist_ok=True)
    ensayo = split != "test"
    for esq in ESQUEMAS:
        if esq_f and esq != esq_f:
            continue
        fuente = None
        pendientes = [("G0", None)] + [(COND[a], a) for a in ARQS]
        for cond, arq in pendientes:
            if arq_f and cond != "G0" and arq != arq_f:
                continue
            if arq_f and cond == "G0":
                continue
            for sem in ([0] if cond == "G0" else SEMILLAS):
                ruta = destino / "parciales" / f"{cond}_{esq}_s{sem}.csv"
                if ruta.exists():
                    print(f"{cond} {esq} s{sem}: ya estaba ({ruta.name}), se salta", flush=True)
                    continue
                if cond != "G0" and fuente is None:
                    fuente = preparar_fuente(esq)
                evaluar(esq, cond, arq, sem, split, umbr, fuente).to_csv(ruta, index=False)
                print(f"{cond} {esq} s{sem}: puntuado ({split})", flush=True)
    compilar(destino, "prueba", ensayo)


def huellas():
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    h = {"git_commit": git("rev-parse", "HEAD"), "git_estado_limpio": git("status", "--porcelain") == "",
         "pesos": {}, "configs": {}, "scores_umbral": {}}
    for esq in ESQUEMAS:
        for arq in ARQS:
            for s in SEMILLAS:
                n = f"{arq}_{esq}_s{s}"
                h["pesos"][n] = sha256(FIN / "modelos" / f"{n}.weights.h5")
                h["scores_umbral"][n] = sha256(FIN / "scores" / f"{n}.npz")
    for c in ("config_final.json", "config_busqueda.json", "config_modelos.json", "config_g0.json",
              "particion_temporal.json", "normalizacion_zscore.json"):
        h["configs"][c] = sha256(Path("configs") / c)
    h["bitacora"] = sha256("bitacora_decisiones_metodologicas.md")
    h["umbrales"] = sha256(OUT / "umbrales.json")
    return h


def cmd_prueba(aviso, reanudar):
    try:
        fecha = datetime.date.fromisoformat(aviso)
    except ValueError:
        sys.exit("--aviso-asesor debe ser una fecha AAAA-MM-DD")
    if fecha > datetime.date.today():
        sys.exit("La fecha del aviso al asesor no puede ser futura.")
    if not (OUT / "umbrales.json").exists():
        sys.exit("Falta results/prueba/umbrales.json: ejecuta primero --umbrales.")
    lock = OUT / "LOCK.json"
    h = huellas()
    if lock.exists():
        previo = _j(lock)
        if not reanudar:
            sys.exit(f"La prueba ya se abrio el {previo['abierta_el']} (estado: {previo['estado']}). "
                     f"No se repite. Si se interrumpio, usa --reanudar.")
        if previo["huellas"] != h:
            sys.exit("Las huellas cambiaron desde que se abrio la prueba (pesos, configuraciones o umbrales). "
                     "No se puede reanudar: revisa results/prueba/LOCK.json.")
        print("Reanudando la prueba ya abierta; se salta lo ya puntuado.")
    else:
        if reanudar:
            sys.exit("No hay prueba abierta que reanudar.")
        if not h["git_estado_limpio"]:
            print("AVISO: hay cambios sin confirmar en git; el commit registrado puede no reflejar todo el codigo.")
        print("Vas a abrir la prueba final (pasos 13593-16992). Es UNA sola vez y queda registrada.")
        if input("Escribe ABRIR PRUEBA para continuar: ").strip() != "ABRIR PRUEBA":
            sys.exit("Cancelado; la prueba no se abrio.")
        OUT.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"estado": "iniciada", "aviso_asesor": aviso,
                                    "abierta_el": datetime.datetime.now().isoformat(timespec="seconds"),
                                    "huellas": h}, indent=2, ensure_ascii=False), encoding="utf-8")
    correr("test", OUT)
    previo = _j(lock)
    previo.update(estado="terminada", terminada_el=datetime.datetime.now().isoformat(timespec="seconds"))
    lock.write_text(json.dumps(previo, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--umbrales", action="store_true")
    g.add_argument("--verificar", action="store_true")
    g.add_argument("--ensayo", action="store_true")
    g.add_argument("--ganadoras", action="store_true")
    g.add_argument("--prueba", action="store_true")
    ap.add_argument("--arq", choices=ARQS)
    ap.add_argument("--esquema", choices=ESQUEMAS)
    ap.add_argument("--semilla", type=int, default=42, choices=SEMILLAS)
    ap.add_argument("--aviso-asesor", dest="aviso")
    ap.add_argument("--reanudar", action="store_true")
    a = ap.parse_args()
    if a.umbrales:
        return cmd_umbrales()
    if a.ganadoras:
        return cmd_ganadoras()
    if a.verificar:
        assert a.arq and a.esquema, "indica --arq y --esquema"
        return cmd_verificar(a.arq, a.esquema, a.semilla)
    if a.ensayo:
        return correr("umbral", OUT_ENSAYO, a.arq, a.esquema)
    if not a.aviso:
        sys.exit("--prueba exige --aviso-asesor AAAA-MM-DD (fecha en que se aviso al asesor).")
    cmd_prueba(a.aviso, a.reanudar)


if __name__ == "__main__":
    main()
