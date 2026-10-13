"""
Semana 5 - Correccion del Esquema B: una serie alterada POR PLIEGUE (H-03).

Problema detectado en semana4_esquema_B.py: los tres pliegues escriben sobre el
mismo arreglo raw_B; los rangos de train se solapan ([0,3398) esta en los 3
train), de modo que lo guardado es la inyeccion del ultimo pliegue
(perfil estimado con [0,10194)) y los conteos reportados para los pliegues 1 y 2
no corresponden a lo guardado. Ademas, con semilla 42 identica, train y val de la
misma longitud reciben las MISMAS posiciones relativas de anomalias.

Esta version (Decision 10):
  - construye un arreglo alterado independiente por pliegue k (train_k y val_k
    inyectados con el perfil estimado SOLO con train_k);
  - construye un arreglo "final": bloque 80 % [0,13593) + prueba [13593,16992),
    ambos con perfil estimado SOLO con el bloque 80 %;
  - semilla = [42, id_pliegue, id_split] (determinista, distinta por bloque).
  - Decision 12: el perfil contextual (P5/P95 hora x dia) se estima POR SENSOR
    (config_esquema_B.json: "perfil_por_sensor": true), como indica el PPI 2.5.2.
    Con false se reproduce el perfil agrupado de la Decision 10.

Requiere: data/processed/pems04_semana2.npz, configs/particion_temporal.json,
          configs/config_esquema_B.json, src/inyeccion_anomalias.py
Genera:   data/processed/esquema_B_pliegue{1,2,3}.npz y esquema_B_final.npz
          (claves: data (T,S,3) alterado, labels (T,S): 0/1 dentro de lo inyectado, -1 fuera,
          tipos (T,S): 0 normal, 1 puntual, 2 contextual, 3 colectiva)
          results/eda/prevalencia_esquema_B_por_pliegue.json
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("src")))
from inyeccion_anomalias import construir_perfil_hora_dia, inyectar_anomalias

DATA = Path("data/processed/pems04_semana2.npz")
PART = Path("configs/particion_temporal.json")
CFG = Path("configs/config_esquema_B.json")
OUT_REP = Path("results/eda/prevalencia_esquema_B_por_pliegue.json")
ID_SPLIT = {"train": 0, "val": 1, "test": 2}


def inyectar_bloque(raw, a, b, perfil, cfg, id_pliegue, split):
    return inyectar_anomalias(
        devolver_tipos=True, raw_split=raw[a:b], idx_inicio_split=a, perfil=perfil,
        prevalencia_objetivo=cfg["prevalencia_objetivo"],
        semilla=[cfg["semilla_inyeccion"], id_pliegue, ID_SPLIT[split]],
        distribucion_tipos=cfg["distribucion_tipos"],
        duraciones_colectivas=cfg["tipos_anomalia"]["colectiva"]["duraciones_pasos"],
        magnitud_sigma=cfg["tipos_anomalia"]["puntual"]["magnitud_sigma"],
        factor_reduccion=cfg["tipos_anomalia"]["colectiva"]["factor_reduccion"],
    )


def construir(raw, cfg, id_pliegue, bloques, train_rango):
    """bloques: lista de (split, a, b). Perfil: SOLO con train_rango."""
    T, S, _ = raw.shape
    alt = raw.copy()
    lab = np.full((T, S), -1, dtype=np.int8)
    tip = np.zeros((T, S), dtype=np.int8)   # 0 normal/fuera, 1 puntual, 2 contextual, 3 colectiva
    por_sensor = bool(cfg.get("perfil_por_sensor", False))
    perfil, _, _ = construir_perfil_hora_dia(raw[train_rango[0]:train_rango[1], :, 0],
                                             por_sensor=por_sensor)
    if por_sensor:
        assert all(np.ndim(v["p5"]) == 1 and v["p5"].shape[0] == S for v in perfil.values()), \
            "el perfil por sensor debe tener un vector (S,) por celda"
    rep = {}
    for split, a, b in bloques:
        x, l, tp = inyectar_bloque(raw, a, b, perfil, cfg, id_pliegue, split)
        alt[a:b], lab[a:b], tip[a:b] = x, l, tp
        assert np.array_equal(tp > 0, l == 1), "tipos y etiquetas no coinciden"
        rep_tipos = {n: int((tp == c).sum()) for n, c in (("puntual", 1), ("contextual", 2), ("colectiva", 3))}
        rep[split] = {"rango": [int(a), int(b)], "n_anomalos": int(l.sum()),
                      "prevalencia": float(l.mean()), "instantes_por_tipo": rep_tipos}
    return alt, lab, rep, tip


def verificar(raw, alt, lab, bloques, nombre):
    T = raw.shape[0]
    fin = max(b for _, _, b in bloques)
    dentro = np.zeros(T, dtype=bool)
    for _, a, b in bloques:
        dentro[a:b] = True
    assert np.array_equal(alt[~dentro], raw[~dentro]), f"{nombre}: se altero algo fuera de los bloques"
    assert (lab[~dentro] == -1).all(), f"{nombre}: etiquetas fuera de los bloques"
    assert np.array_equal(alt[:fin][lab[:fin] == 0], raw[:fin][lab[:fin] == 0]), \
        f"{nombre}: hay features alteradas con label 0"
    for sp, a, b in bloques:
        p = lab[a:b].mean()
        assert 0.029 <= p <= 0.032, f"{nombre}/{sp}: prevalencia {p:.4f} fuera de [2.9 %, 3.2 %]"
    tr = [x for x in bloques if x[0] == "train"][0]
    va = [x for x in bloques if x[0] in ("val", "test")][0]
    n = min(tr[2] - tr[1], va[2] - va[1])
    assert not np.array_equal(lab[tr[1]:tr[1] + n], lab[va[1]:va[1] + n]), \
        f"{nombre}: train y val tienen posiciones identicas de anomalias"


def main():
    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    part = json.loads(PART.read_text(encoding="utf-8-sig"))
    raw = np.load(DATA)["data"]
    reporte = {"semilla_base": cfg["semilla_inyeccion"],
               "perfil_por_sensor": bool(cfg.get("perfil_por_sensor", False)),
               "esquema_semillas": "[semilla_base, id_pliegue (0=final), id_split (0 train,1 val,2 test)]",
               "pliegues": {}}

    for pl in part["pliegues"]:
        k = pl["pliegue"]
        (t0, t1), (v0, v1) = pl["train_idx"], pl["val_idx"]
        bl = [("train", t0, t1), ("val", v0, v1)]
        alt, lab, rep, tip = construir(raw, cfg, k, bl, (t0, t1))
        verificar(raw, alt, lab, bl, f"pliegue{k}")
        np.savez_compressed(f"data/processed/esquema_B_pliegue{k}.npz", data=alt, labels=lab, tipos=tip)
        reporte["pliegues"][f"pliegue{k}"] = rep
        print(f"Pliegue {k}: " + " | ".join(f"{s} {r['prevalencia']*100:.3f}% ({r['n_anomalos']:,})"
                                             for s, r in rep.items()))

    te0, te1 = part["test_idx"]
    bl = [("train", 0, te0), ("test", te0, te1)]
    alt, lab, rep, tip = construir(raw, cfg, 0, bl, (0, te0))
    verificar(raw, alt, lab, bl, "final")
    np.savez_compressed("data/processed/esquema_B_final.npz", data=alt, labels=lab, tipos=tip)
    reporte["pliegues"]["final"] = rep
    print("Final  : " + " | ".join(f"{s} {r['prevalencia']*100:.3f}% ({r['n_anomalos']:,})" for s, r in rep.items()))

    # Reproducibilidad: segunda corrida del pliegue 1 debe ser identica
    t0, t1 = part["pliegues"][0]["train_idx"]; v0, v1 = part["pliegues"][0]["val_idx"]
    alt2, lab2, _, _ = construir(raw, cfg, 1, [("train", t0, t1), ("val", v0, v1)], (t0, t1))
    assert np.array_equal(lab2, np.load("data/processed/esquema_B_pliegue1.npz")["labels"]), "no reproducible"
    print("Controles OK: nada fuera de los bloques, label 0 = sin cambios, prevalencia ~3 %,"
          " train != val, reproducible.")

    OUT_REP.parent.mkdir(parents=True, exist_ok=True)
    OUT_REP.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte: {OUT_REP}")


if __name__ == "__main__":
    main()
