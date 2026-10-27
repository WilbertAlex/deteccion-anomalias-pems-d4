"""
Semana 6 - Cronometro de una epoca (hallazgos H-01 y H-02 de la revision del 13 oct.).

Mide el tiempo REAL por paso de entrenamiento de un modelo recurrente (G1 por defecto) sobre el
pliegue elegido, con el cargador de src/datos.py, y extrapola:
  - una epoca completa del pliegue 3 (pasos = ceil(n_train / lote));
  - el costo de la busqueda del PPI 2.7.1 (12 configs x 3 pliegues x 3 semillas x 6 arquitecturas
    x 2 esquemas = 1 296 entrenamientos + 36 reentrenamientos finales) para un numero de epocas
    por entrenamiento que TU eliges como escenario (--epocas-escenario; no es un dato medido).
La extrapolacion supone que todas las arquitecturas cuestan lo mismo que la medida y que los pasos por
epoca de cada pliegue son proporcionales al largo de su tramo de entrenamiento; son cotas de orden de magnitud.
El modelo por defecto NO es la configuracion del Cuadro 6 (LSTM 64, 1 capa): pasa la tuya.

Uso:  python semana6_cronometro_epoca.py [--esquema B] [--pliegue 3] [--pasos 300] [--calentamiento 30]
                                           [--lote 64] [--tipo lstm] [--unidades 64] [--capas 1]
                                           [--epocas-escenario 30] [--etiqueta cpu-windows]
Genera: results/eda/cronometro_epoca_<etiqueta>.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esquema", default="B", choices=["A", "B"])
    ap.add_argument("--pliegue", default="3")
    ap.add_argument("--pasos", type=int, default=300)
    ap.add_argument("--calentamiento", type=int, default=30)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--tipo", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--unidades", type=int, default=64)
    ap.add_argument("--capas", type=int, default=1)
    ap.add_argument("--epocas-escenario", type=int, default=30)
    ap.add_argument("--etiqueta", default="local", help="nombre del entorno medido, p. ej. cpu-windows o gpu-wsl2")
    a = ap.parse_args()
    pliegue = a.pliegue if a.pliegue == "final" else int(a.pliegue)

    import numpy as np
    import tensorflow as tf
    from datos import Fuente, dataset, rangos
    from entrenar import fijar_semillas, dispositivos, construir_modelo_recurrente

    fijar_semillas(42)
    disp = dispositivos()
    print("Dispositivos:", disp)
    f = Fuente(a.esquema, pliegue)
    ds, n = dataset(f, "train", batch=a.lote, semilla=42)
    pasos_epoca = math.ceil(n / a.lote)
    print(f"Esquema {a.esquema}, pliegue {pliegue}: {n:,} ventanas de entrenamiento -> {pasos_epoca:,} pasos por epoca (lote {a.lote})")

    m = construir_modelo_recurrente(a.tipo, a.unidades, a.capas)
    it = iter(ds)
    # calentamiento (incluye la construccion del grafo)
    t0 = time.perf_counter()
    for _ in range(a.calentamiento):
        x, y = next(it)
        m.train_on_batch(x, y)
    t_calent = time.perf_counter() - t0
    # medicion
    t_datos, t_total = 0.0, 0.0
    ts = []
    for _ in range(a.pasos):
        t1 = time.perf_counter()
        x, y = next(it)
        t2 = time.perf_counter()
        m.train_on_batch(x, y)
        t3 = time.perf_counter()
        t_datos += t2 - t1
        ts.append(t3 - t1)
    ts = np.array(ts)
    paso_med, paso_p50 = float(ts.mean()), float(np.median(ts))
    t_epoca = paso_med * pasos_epoca
    # largo del tramo de entrenamiento de cada pliegue (los pasos por epoca son proporcionales)
    fin = lambda p: rangos(p)["train"][1]
    ent_por_pliegue = 12 * 3 * 6 * 2        # 12 configs x 3 semillas x 6 arquitecturas x 2 esquemas
    t_epoca_pl = {p: t_epoca * fin(p) / fin(pliegue) for p in (1, 2, 3, "final")}
    t_busq = sum(ent_por_pliegue * a.epocas_escenario * t_epoca_pl[p] for p in (1, 2, 3))
    t_final = 6 * 3 * 2 * a.epocas_escenario * t_epoca_pl["final"]      # 36 reentrenamientos con el bloque 80 %
    res = {"etiqueta": a.etiqueta, "dispositivos": disp, "esquema": a.esquema, "pliegue": str(pliegue), "ventanas_train": int(n),
           "lote": a.lote, "pasos_por_epoca": pasos_epoca,
           "modelo_medido": {"tipo": a.tipo, "unidades": a.unidades, "capas": a.capas,
                             "nota": "configuracion de MEDICION; no es la del Cuadro 6"},
           "pasos_medidos": a.pasos, "calentamiento_s": t_calent,
           "segundos_por_paso_media": paso_med, "segundos_por_paso_mediana": paso_p50,
           "fraccion_tiempo_en_datos": t_datos / ts.sum(),
           "epoca_completa_estimada_min": t_epoca / 60,
           "epoca_estimada_min_por_pliegue": {str(p): t_epoca_pl[p] / 60 for p in (1, 2, 3, "final")},
           "escenario": {"epocas_por_entrenamiento": a.epocas_escenario,
                         "entrenamientos_busqueda": ent_por_pliegue * 3, "reentrenamientos_finales": 36,
                         "horas_busqueda": t_busq / 3600, "horas_reentrenamiento_final": t_final / 3600,
                         "dias_totales_24h": (t_busq + t_final) / 86400,
                         "nota": "extrapolacion de orden de magnitud; supone costo igual en las 6 arquitecturas"}}
    print(f"\nPaso medio: {paso_med * 1000:.1f} ms (mediana {paso_p50 * 1000:.1f} ms); "
          f"{res['fraccion_tiempo_en_datos'] * 100:.0f} % del tiempo es carga de datos")
    print(f"Epoca completa estimada (pliegue {pliegue}): {res['epoca_completa_estimada_min']:.1f} min")
    print(f"Escenario de {a.epocas_escenario} epocas por entrenamiento: busqueda {res['escenario']['horas_busqueda']:.0f} h "
          f"+ reentrenamiento final {res['escenario']['horas_reentrenamiento_final']:.0f} h "
          f"= {res['escenario']['dias_totales_24h']:.1f} dias de computo continuo")
    Path("results/eda").mkdir(parents=True, exist_ok=True)
    out = Path(f"results/eda/cronometro_epoca_{a.etiqueta}.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
