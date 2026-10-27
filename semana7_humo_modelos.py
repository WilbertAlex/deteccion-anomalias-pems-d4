"""
Semana 7 - Prueba de humo de G1-G6 (src/modelos.py). Cada control es un assert.
Usa configuraciones DIMINUTAS de prueba (NO son las del Cuadro 6) y un lote real del Esquema B, pliegue 1.
Comprueba: forma de salida, float32, valores finitos, rango [0,1] en los supervisados, que un paso de
entrenamiento baje o mantenga la perdida finita, que falte-un-hiperparametro falle, y que el submuestreo
fijo (paso_t) sea determinista y subconjunto del completo.
Genera: results/eda/humo_modelos.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import tensorflow as tf
from datos import Fuente, dataset
from entrenar import fijar_semillas
from modelos import ARQUITECTURAS, construir, score_ae
from semana7_busqueda import muestrear

PRUEBA = {  # SOLO formas; no son del Cuadro 6
    "lstm": dict(unidades=[4], dropout=0.0, lr=1e-3),
    "gru": dict(unidades=[4, 3], dropout=0.1, lr=1e-3),
    "cnn1d": dict(filtros=[4, 4], kernel=3, weight_decay=1e-4, lr=1e-3),
    "transformer": dict(bloques=1, cabezas=2, d_model=8, ff=8, dropout=0.1, lr=1e-3, weight_decay=1e-4),
    "cnn_lstm_gru": dict(filtros=4, kernel=3, unidades_lstm=4, unidades_gru=4, dropout=0.0, lr=1e-3),
    "autoencoder": dict(capas=[8, 4, 3], l2=0.0, lr=1e-3),
}
ok = n = 0


def chk(c, msg):
    global ok, n
    n += 1
    assert c, "FALLO: " + msg
    ok += 1
    print(f"  [OK ] {msg}")


fijar_semillas(0)
f = Fuente("B", 1)
ds, ntr = dataset(f, "train", batch=32, semilla=0)
x, y = next(iter(ds))
res = {}
for arq in ARQUITECTURAS:
    print(arq)
    m = construir(arq, PRUEBA[arq])
    chk(m.input_shape == (None, 12, 8), f"{arq}: entrada (None,12,8)")
    chk(m.count_params() > 0, f"{arq}: parametros entrenables = {m.count_params()}")
    if arq == "autoencoder":
        chk(m.output_shape == (None, 12, 8), "autoencoder: salida (None,12,8)")
        s = score_ae(m, x).numpy()
        chk(s.shape == (32,) and np.isfinite(s).all() and (s >= 0).all(), "autoencoder: score MSE (32,) finito y >= 0")
        l0 = m.evaluate(x, x, verbose=0)
        for _ in range(20):
            m.train_on_batch(x, x)
        l1 = m.evaluate(x, x, verbose=0)
        chk(np.isfinite(l1) and l1 < l0, f"autoencoder: 20 pasos sobre un lote bajan la perdida ({l0:.4f} -> {l1:.4f})")
    else:
        p = m(x, training=False).numpy()
        chk(p.shape == (32, 1) and p.dtype == np.float32, f"{arq}: salida (32,1) float32")
        chk(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), f"{arq}: probabilidades finitas en [0,1]")
        l0 = m.evaluate(x, y, verbose=0)[0]
        for _ in range(20):
            m.train_on_batch(x, y)
        l1 = m.evaluate(x, y, verbose=0)[0]
        chk(np.isfinite(l1) and l1 < l0, f"{arq}: 20 pasos sobre un lote bajan la perdida ({l0:.4f} -> {l1:.4f})")
    res[arq] = {"parametros": int(m.count_params()), "config_de_prueba": PRUEBA[arq],
                "nota": "configuracion de prueba de forma; NO es la del Cuadro 6"}
    bad = dict(PRUEBA[arq]); bad.pop("lr")
    try:
        construir(arq, bad); fallo = False
    except KeyError:
        fallo = True
    chk(fallo, f"{arq}: sin hiperparametro obligatorio lanza KeyError (no hay valores por defecto)")
print("submuestreo fijo")
t1, s1, y1 = f.indices("train")
t4, s4, y4 = f.indices("train", 4)
t4b, s4b, _ = f.indices("train", 4)
chk(np.array_equal(t4, t4b) and np.array_equal(s4, s4b), "paso_t=4 es determinista")
chk(np.all(t4 % 4 == 0) and len(t4) < len(t1), f"paso_t=4 deja {len(t4):,} de {len(t1):,} ventanas")
conj = set(zip(t1.tolist(), s1.tolist()))
chk(all((a, b) in conj for a, b in zip(t4[:5000].tolist(), s4[:5000].tolist())), "paso_t=4 es subconjunto del completo")
chk(np.array_equal(y4, f.labels[t4, s4].astype(np.float32)), "paso_t=4 conserva las etiquetas correspondientes")
print("Cuadro 6 (configuracion inicial) y espacio de busqueda")
CM = json.loads(Path("configs/config_modelos.json").read_text(encoding="utf-8-sig"))
L = tf.keras.layers
def capas(m, tipo):
    return [l for l in m.layers if isinstance(l, tipo)]
ini = {a: construir(a, CM["inicial"][a]) for a in ARQUITECTURAS}
for a in ("lstm", "gru"):
    tipo = L.LSTM if a == "lstm" else L.GRU
    chk([l.units for l in capas(ini[a], tipo)] == [64, 32] and ini[a].layers[-1].units == 1
        and ini[a].layers[-1].activation.__name__ == "sigmoid", f"{a}: capas {a.upper()} 64 y 32 + densa sigmoide (Cuadro 6)")
chk([l.filters for l in capas(ini["cnn1d"], L.Conv1D)] == [32, 64, 128] and all(l.kernel_size == (3,) for l in capas(ini["cnn1d"], L.Conv1D))
    and len(capas(ini["cnn1d"], L.GlobalMaxPooling1D)) == 1, "cnn1d: Conv1D 32/64/128, kernel 3, GlobalMaxPooling1D (Cuadro 6)")
dn = [l.units for l in capas(ini["autoencoder"], L.Dense)]
chk(dn == [32, 16, 8, 16, 32, 96] and ini["autoencoder"].layers[1].output.shape[-1] == 96,
    "autoencoder: Flatten 96 -> 32,16,8 -> simetrico -> 96 (Cuadro 6)")
mt = ini["transformer"]
chk(len(capas(mt, L.MultiHeadAttention)) == 2 and capas(mt, L.MultiHeadAttention)[0].num_heads == 4
    and 128 in [l.units for l in capas(mt, L.Dense)] and 64 in [l.units for l in capas(mt, L.Dense)],
    "transformer: 2 bloques, 4 cabezas, embedding 64, feed-forward 128 (Cuadro 6)")
g6 = ini["cnn_lstm_gru"]
chk(capas(g6, L.Conv1D)[0].filters == 64 and capas(g6, L.LSTM)[0].units == 64 and capas(g6, L.GRU)[0].units == 32,
    "cnn_lstm_gru: Conv1D 64, LSTM 64, GRU 32 (Cuadro 6)")
opt = {a: type(ini[a].optimizer).__name__ for a in ARQUITECTURAS}
chk(opt == {"lstm": "Adam", "gru": "Adam", "cnn1d": "AdamW", "autoencoder": "Adam", "transformer": "AdamW", "cnn_lstm_gru": "Adam"},
    f"optimizadores del Cuadro 6: {opt}")
chk(abs(float(ini["transformer"].optimizer.learning_rate) - 5e-4) < 1e-9 and abs(float(ini["lstm"].optimizer.learning_rate) - 1e-3) < 1e-9,
    "tasas de aprendizaje: Transformer 5e-4, resto 1e-3")
for a in ARQUITECTURAS:
    cfgs = muestrear(CM["espacio"][a], 12, 2026, inicial=CM["inicial"][a])
    claves = set(CM["inicial"][a])
    chk(len(cfgs) == 12 and cfgs[0] == CM["inicial"][a] and len({json.dumps(c, sort_keys=True) for c in cfgs}) == 12
        and all(set(c) == claves for c in cfgs) and cfgs == muestrear(CM["espacio"][a], 12, 2026, inicial=CM["inicial"][a]),
        f"{a}: 12 configuraciones distintas, la 0 es la del Cuadro 6, determinista")
    for c in cfgs[1:3]:
        construir(a, c)
    res[a]["parametros_cuadro6"] = int(ini[a].count_params())
print(f"\nResultado: {ok}/{n} controles OK")
Path("results/eda").mkdir(parents=True, exist_ok=True)
Path("results/eda/humo_modelos.json").write_text(json.dumps(
    {"controles_ok": ok, "controles_total": n, "tensorflow": tf.__version__, "modelos": res}, indent=2), encoding="utf-8")
assert ok == n
