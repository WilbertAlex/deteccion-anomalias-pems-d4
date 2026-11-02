"""
Semana 9 - Presupuesto de computo a partir de los cronometros con fit() (hallazgo H-04 de la revision del 2 nov).

Lee results/eda/cronometro_fit_*.json (uno por arquitectura, generados por semana9_cronometro_fit.py), toma
el costo por paso de fit() con steps_per_execution = 1 (la forma en que entrena la busqueda real) y calcula
las horas de entrenamiento segun la reduccion de las Decisiones 15-16. Todo se calcula de los archivos; nada
se escribe a mano:
  - Costo de un pliegue de la busqueda: ms/paso * pasos/epoca del pliegue 3 con paso_t = 4, escalado a los
    pliegues 1, 2 y 3 por el tamano de su bloque de entrenamiento (configs/particion_temporal.json).
  - Reentrenamiento final: todas las ventanas del bloque del 80 % (tamano relativo al pliegue 3 tomado de
    la particion: 13 593 / 10 194 pasos), 3 semillas, 2 esquemas, hasta N epocas.
  - Ablacion de canales (Decision 15) y pesos de clase: UN entrenamiento final por arquitectura (ablacion,
    solo arquitecturas supervisadas) y por arquitectura x esquema (pesos de clase). Es la lectura de los
    conteos de la Decision 16 (5 y 12 entrenamientos); si el equipo los define distinto, se cambian los
    argumentos --abl-entrenamientos y --pesos-entrenamientos.

NO incluye: validacion por epoca, puntuacion final, G7, ni arquitecturas sin cronometrar. Para las que
faltan se dan dos cotas: igual al promedio de las medidas y el doble de ese promedio.

Uso (desde la raiz del proyecto):
  python semana9_presupuesto.py
Genera: results/eda/presupuesto_computo.json y .md (sobrescribe; es un calculo, no una medicion).
"""
import argparse
import glob
import json
from pathlib import Path

ARQ = ["lstm", "gru", "cnn1d", "autoencoder", "transformer", "cnn_lstm_gru"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=12)
    ap.add_argument("--epocas", type=int, default=10, help="epocas maximas (cota superior; la parada temprana solo baja)")
    ap.add_argument("--semillas-final", type=int, default=3)
    ap.add_argument("--esquemas", type=int, default=2)
    ap.add_argument("--abl-entrenamientos", type=int, default=5)
    ap.add_argument("--pesos-entrenamientos", type=int, default=12)
    ap.add_argument("--horas-disponibles", type=float, default=648, help="horas restantes por maquina hasta el 29 nov")
    a = ap.parse_args()

    part = json.loads(Path("configs/particion_temporal.json").read_text(encoding="utf-8-sig"))
    tam = [p["train_idx"][1] - p["train_idx"][0] for p in part["pliegues"]]
    suma_pliegues = sum(t / tam[2] for t in tam)  # 1/3 + 2/3 + 1 = 2.0
    f80 = part["bloques_pre_test"][-1]["idx"][1] / tam[2]  # bloque 80 % respecto al pliegue 3

    medidas = {}
    for f in sorted(glob.glob("results/eda/cronometro_fit_*.json")):
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        m = j["meta"]
        if m["esquema"] != "B" or str(m["pliegue"]) != "3" or m["paso_t"] != 4 or m["lote"] != 64:
            continue
        r = j["resultados"].get("fit_spe1")
        if r is None:
            continue
        medidas[m["arquitectura"]] = {"archivo": f, "ms_paso": r["ms_por_paso"],
                                      "epoca_busqueda_min": r["epoca_busqueda_min"], "epoca_final_pliegue3_min": r["epoca_final_min"]}

    filas = {}
    for arq, d in medidas.items():
        busq = a.configs * a.epocas * d["epoca_busqueda_min"] * suma_pliegues * a.esquemas / 60
        ep80 = d["epoca_final_pliegue3_min"] * f80
        final = a.semillas_final * a.esquemas * a.epocas * ep80 / 60
        filas[arq] = {**d, "epoca_final_80_min": ep80, "busqueda_h": busq, "final_h": final,
                      "ablacion_h": (a.epocas * ep80 / 60) * (1 if arq != "autoencoder" else 0) * 1.0,
                      "pesos_clase_h": (a.pesos_entrenamientos / len(ARQ)) * a.epocas * ep80 / 60}
    falta = [x for x in ARQ if x not in filas]
    if not filas:
        raise SystemExit("No hay cronometros validos (esquema B, pliegue 3, paso_t 4, lote 64) en results/eda/.")
    n_med = len(filas)
    tot = lambda clave: sum(v[clave] for v in filas.values())
    medio = {k: tot(k) / n_med for k in ("busqueda_h", "final_h", "ablacion_h", "pesos_clase_h")}
    # arquitecturas faltantes: costo medio (cota baja) o doble (cota alta)
    def total(mult):
        t = tot("busqueda_h") + tot("final_h") + tot("pesos_clase_h") + tot("ablacion_h")
        t += mult * len(falta) * (medio["busqueda_h"] + medio["final_h"] + medio["pesos_clase_h"])
        t += mult * sum(1 for x in falta if x != "autoencoder") * medio["ablacion_h"]
        return t
    res = {"supuestos": vars(a), "factor_pliegues": suma_pliegues, "factor_bloque80": f80,
           "medidas": filas, "sin_medir": falta,
           "total_h_si_faltantes_cuestan_el_promedio": total(1.0), "total_h_si_faltantes_cuestan_el_doble": total(2.0),
           "dias_continuos": [total(1.0) / 24, total(2.0) / 24], "horas_disponibles_por_maquina": a.horas_disponibles}
    L = ["# Presupuesto de computo (calculado de results/eda/cronometro_fit_*.json)", "",
         f"Supuestos: {a.configs} configuraciones, hasta {a.epocas} epocas, {a.semillas_final} semillas en el reentrenamiento final, "
         f"{a.esquemas} esquemas. Factor de los 3 pliegues = {suma_pliegues:.2f}; bloque 80 % = {f80:.3f} x pliegue 3.", "",
         "| Arquitectura | ms/paso | min/epoca busqueda | min/epoca final (80 %) | busqueda (h) | reent. final (h) | ablacion (h) | pesos de clase (h) |",
         "|---|---|---|---|---|---|---|---|"]
    for arq, v in filas.items():
        L.append(f"| {arq} | {v['ms_paso']:.2f} | {v['epoca_busqueda_min']:.2f} | {v['epoca_final_80_min']:.2f} | {v['busqueda_h']:.1f} | "
                 f"{v['final_h']:.1f} | {v['ablacion_h']:.1f} | {v['pesos_clase_h']:.1f} |")
    L += ["", f"Sin medir: {', '.join(falta) if falta else 'ninguna'}.",
          f"Total si las faltantes cuestan el promedio de las medidas: **{total(1.0):.0f} h = {total(1.0) / 24:.1f} dias continuos**.",
          f"Total si cuestan el doble: **{total(2.0):.0f} h = {total(2.0) / 24:.1f} dias**. Disponible por maquina hasta el 29 nov: {a.horas_disponibles:.0f} h.",
          "", "No incluye validacion por epoca, puntuacion final ni G7. La parada temprana solo reduce estas cifras."]
    out = Path("results/eda/presupuesto_computo")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
