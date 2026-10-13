"""
Semana 5 - Validacion tecnica del pipeline (gate de fuga temporal, parte 1).

Pruebas ACTIVAS (se perturba el futuro y se comprueba que el pasado no cambia),
no solo comparaciones de indices. Cada control es un assert; si uno falla, el
script termina con error y el pipeline NO esta listo para la bateria.

Requiere: data/processed/{pems04_semana2,labels_esquema_A,esquema_B_pliegue1..3,esquema_B_final}.npz
          configs/{particion_temporal,normalizacion_zscore}.json, src/{ventanas,g0_linea_base}.py
Genera:   results/eda/validacion_pipeline.json
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path("src")))
from ventanas import construir_ventanas
from g0_linea_base import scores_g0

P = Path("data/processed")
R = []


def chk(nombre, cond, detalle=""):
    R.append({"check": nombre, "ok": bool(cond), "detalle": detalle})
    print(f"  [{'OK ' if cond else 'FAIL'}] {nombre}" + (f"  ({detalle})" if detalle else ""))


def etiquetas_A_causales(flow, w=288, umbral=3.0):
    df = pd.DataFrame(flow)
    mu = df.rolling(w, min_periods=w).mean().shift(1).to_numpy()
    sd = df.rolling(w, min_periods=w).std(ddof=0).shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (flow - mu) / sd
    lab = np.where(np.isnan(z), -1, (np.abs(z) > umbral).astype(int))
    return np.where(np.isnan(sd) | (sd == 0), -1, lab)


def main():
    part = json.loads(Path("configs/particion_temporal.json").read_text(encoding="utf-8"))
    norm = json.loads(Path("configs/normalizacion_zscore.json").read_text(encoding="utf-8"))
    raw = np.load(P / "pems04_semana2.npz")["data"]
    flow = raw[:, :, 0]
    lab_A = np.load(P / "labels_esquema_A.npz")["labels"]
    te0, te1 = part["test_idx"]

    print("1) Particion")
    for pl in part["pliegues"]:
        k = pl["pliegue"]
        chk(f"P{k}: train < val < prueba", pl["train_idx"][1] <= pl["val_idx"][0] and pl["val_idx"][1] <= te0)

    print("2) Normalizacion solo con entrenamiento")
    for pl, n in zip(part["pliegues"], norm["pliegues"]):
        a, b = pl["train_idx"]
        for i, v in enumerate(["flow", "occupancy", "speed"]):
            ok = abs(n["variables"][v]["mean"] - raw[a:b, :, i].mean()) < 1e-6 and \
                 abs(n["variables"][v]["std"] - raw[a:b, :, i].std()) < 1e-6
            chk(f"P{pl['pliegue']} {v}: mu/sigma = solo train", ok)
    chk("Existe entrada 'final' (0..inicio prueba)", "final" in norm)
    if "final" in norm:
        fin = norm["final"]
        rango = fin.get("rango_pasos", fin.get("train_idx"))
        chk("'final' cubre exactamente [0, inicio de prueba)", list(rango) == [0, te0], f"rango={rango}")
        chk("'final' mu(flujo) = bloque 80 %", abs(fin["variables"]["flow"]["mean"] - raw[:te0, :, 0].mean()) < 1e-6)

    print("3) Esquema A es causal (prueba de perturbacion)")
    t0 = 9000
    f2 = flow.copy(); f2[t0 + 1:] = np.random.default_rng(0).uniform(0, 900, f2[t0 + 1:].shape)
    la, lb = etiquetas_A_causales(flow), etiquetas_A_causales(f2)
    chk("Alterar flujo futuro NO cambia etiquetas A hasta t0", np.array_equal(la[:t0 + 1], lb[:t0 + 1]))
    chk("Etiquetas A guardadas = recalculo independiente", np.array_equal(lab_A, la))
    chk("A: primeros 288 pasos sin etiqueta", (lab_A[:288] == -1).all())
    chk("A: despues solo 0/1 (salvo sigma=0)", set(np.unique(lab_A[288:])) <= {-1, 0, 1})

    print("4) G0 es causal (prueba de perturbacion)")
    for w in (6, 11, 22, 44):
        s1, s2 = scores_g0(flow, w), scores_g0(f2, w)
        chk(f"G0 w={w}: futuro alterado no cambia scores hasta t0",
            np.allclose(s1[:t0 + 1], s2[:t0 + 1], equal_nan=True))
        chk(f"G0 w={w}: el paso t no entra en su referencia",
            np.isnan(s1[:w]).all())

    print("5) Esquema B por pliegue")
    for pl in part["pliegues"]:
        k = pl["pliegue"]
        z = np.load(P / f"esquema_B_pliegue{k}.npz"); alt, lab = z["data"], z["labels"]
        v1 = pl["val_idx"][1]
        chk(f"B P{k}: nada alterado despues de val_end", np.array_equal(alt[v1:], raw[v1:]))
        chk(f"B P{k}: sin etiquetas despues de val_end", (lab[v1:] == -1).all())
        chk(f"B P{k}: label 0 => features intactas", np.array_equal(alt[:v1][lab[:v1] == 0], raw[:v1][lab[:v1] == 0]))
        chk(f"B P{k}: label 1 altera >=99 % de los flujos",
            (alt[:v1, :, 0][lab[:v1] == 1] != flow[:v1][lab[:v1] == 1]).mean() >= 0.99,
            f"{(alt[:v1, :, 0][lab[:v1] == 1] != flow[:v1][lab[:v1] == 1]).mean():.4f}")
        tr0, tr1 = pl["train_idx"]; va0, va1 = pl["val_idx"]; n = min(tr1 - tr0, va1 - va0)
        chk(f"B P{k}: posiciones train != val", not np.array_equal(lab[tr0:tr0 + n], lab[va0:va0 + n]))
    z = np.load(P / "esquema_B_final.npz")
    chk("B final: prevalencia prueba en [2.9 %, 3.2 %]", 0.029 <= z["labels"][te0:te1].mean() <= 0.032)
    chk("B final: nada fuera de [0, fin)", np.array_equal(z["data"][te1:], raw[te1:]))

    print("6) Ventanas no miran el futuro")
    for pl in part["pliegues"]:
        a, b = pl["train_idx"]
        X, meta = construir_ventanas(raw.astype(np.float32), a, b)
        chk(f"P{pl['pliegue']}: t_max de ventanas de train < train_end", meta[:, 1].max() < b)
        chk(f"P{pl['pliegue']}: ventana termina en t (ultimo paso = dato de t)",
            np.array_equal(X[-1], raw[meta[-1, 1] - 11: meta[-1, 1] + 1, meta[-1, 0], :].astype(np.float32)))
        del X

    ok = sum(r["ok"] for r in R)
    print(f"\nResultado: {ok}/{len(R)} controles OK")
    Path("results/eda").mkdir(parents=True, exist_ok=True)
    Path("results/eda/validacion_pipeline.json").write_text(
        json.dumps({"n_ok": ok, "n_total": len(R), "todo_ok": ok == len(R), "checks": R}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    assert ok == len(R), "Hay controles fallidos: NO avanzar a la bateria"


if __name__ == "__main__":
    main()
