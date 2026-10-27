"""
src/datos.py - Cargador comun de datos para G1-G6 (y, con adaptacion, G7).

Arma, por esquema (A/B) y pliegue (1, 2, 3 o "final"), los conjuntos train/val/test como
tf.data.Dataset de ventanas (batch, 12, 8) en float32, SIN materializar las ventanas:
solo se guardan los indices (t, sensor, etiqueta) y las ventanas se extraen al vuelo
(tf.gather_nd) del arreglo normalizado (T, S, 3) mas las 5 variables temporales (T, 5).

Reglas (todas verificadas por semana6_verificar_datos.py):
  - Ventana de 12 pasos: t-11 .. t, etiqueta en t (PPI 2.2); features = flujo, ocupacion,
    velocidad (z-score) + hour_sin, hour_cos, day_sin, day_cos, is_peak_hour = 8.
  - z-score con mu/sigma del ENTRENAMIENTO del pliegue (configs/normalizacion_zscore.json;
    entrada "final" para el bloque 80 %). Bajo el Esquema B tambien se usan las estadisticas
    de la serie ORIGINAL (Decision 14).
  - Esquema A: serie original + labels_esquema_A.npz. Esquema B: arreglo alterado del pliegue
    (esquema_B_pliegue{k}.npz) o del bloque final (esquema_B_final.npz).
  - Solo se usan instantes con etiqueta 0/1 (se descarta -1) y t >= 11.
  - float32 en todo el pipeline; lotes de 64 por defecto (Cuadro 6).
"""
import json
from pathlib import Path

import numpy as np

VENTANA = 12
N_FEATURES = 8
P = Path("data/processed")
CFG = Path("configs")
SPLITS_VALIDOS = ("train", "val", "test")


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def rangos(pliegue):
    """Devuelve {split: (a, b)} para un pliegue (1, 2, 3) o 'final'."""
    part = _json(CFG / "particion_temporal.json")
    if pliegue == "final":
        te0, te1 = part["test_idx"]
        return {"train": (0, te0), "test": (te0, te1)}
    pl = next(x for x in part["pliegues"] if x["pliegue"] == pliegue)
    return {"train": tuple(pl["train_idx"]), "val": tuple(pl["val_idx"])}


def _stats(pliegue):
    norm = _json(CFG / "normalizacion_zscore.json")
    if pliegue == "final":
        v = norm["final"]["variables"]
    else:
        v = next(x for x in norm["pliegues"] if x["pliegue"] == pliegue)["variables"]
    return np.array([[v[n]["mean"], v[n]["std"]] for n in ("flow", "occupancy", "speed")], dtype=np.float64)


class Fuente:
    """Datos normalizados de un (esquema, pliegue): X3 (T,S,3) float32, temporales (T,5), labels (T,S)."""

    def __init__(self, esquema, pliegue):
        assert esquema in ("A", "B") and pliegue in (1, 2, 3, "final")
        self.esquema, self.pliegue = esquema, pliegue
        base = np.load(P / "pems04_semana2.npz")
        if esquema == "A":
            raw = base["data"]
            self.labels = np.load(P / "labels_esquema_A.npz")["labels"]
        else:
            nombre = "esquema_B_final.npz" if pliegue == "final" else f"esquema_B_pliegue{pliegue}.npz"
            z = np.load(P / nombre)
            raw, self.labels = z["data"], z["labels"]
        st = _stats(pliegue)
        self.X3 = ((raw - st[:, 0]) / st[:, 1]).astype(np.float32)           # (T, S, 3)
        self.temp = np.stack([base["hour_sin"], base["hour_cos"], base["day_sin"],
                              base["day_cos"], base["is_peak_hour"]], axis=1).astype(np.float32)  # (T, 5)
        self.T, self.S = self.X3.shape[:2]
        self.rangos = rangos(pliegue)

    def indices(self, split, paso_t=1):
        """(t, s, y) de los instantes con etiqueta valida en el split y t >= VENTANA-1.
        paso_t > 1: submuestra FIJA (determinista, sin azar) que conserva solo los instantes t
        multiplos de paso_t; se usa unicamente en la fase de busqueda de hiperparametros."""
        a, b = self.rangos[split]
        a = max(a, VENTANA - 1)
        sub = self.labels[a:b]
        tt, ss = np.nonzero(sub != -1)
        tt = tt + a
        if paso_t > 1:
            k = (tt % paso_t) == 0
            tt, ss = tt[k], ss[k]
        return tt.astype(np.int32), ss.astype(np.int32), self.labels[tt, ss].astype(np.float32)

    def ventanas_numpy(self, t, s):
        """Ventanas (n, 12, 8) float32 para pares (t, s). Solo para verificacion y pruebas pequenas."""
        offs = np.arange(-(VENTANA - 1), 1)
        tt = t[:, None] + offs[None, :]
        a = self.X3[tt, s[:, None], :]
        b = self.temp[tt]
        return np.concatenate([a, b], axis=2)


def dataset(fuente, split, batch=64, barajar=None, semilla=0, paso_t=1):
    """tf.data.Dataset de (ventanas (B,12,8), y (B,)). barajar=None -> True solo en train."""
    import tensorflow as tf
    assert split in fuente.rangos, f"el pliegue {fuente.pliegue} no tiene split '{split}'"
    t, s, y = fuente.indices(split, paso_t)
    barajar = (split == "train") if barajar is None else barajar
    X3 = tf.constant(fuente.X3)
    TMP = tf.constant(fuente.temp)
    offs = tf.range(-(VENTANA - 1), 1, dtype=tf.int32)

    def extraer(tb, sb, yb):
        tt = tb[:, None] + offs[None, :]                                  # (B, 12)
        ss = tf.broadcast_to(sb[:, None], tf.shape(tt))
        a = tf.gather_nd(X3, tf.stack([tt, ss], axis=-1))                 # (B, 12, 3)
        b = tf.gather(TMP, tt)                                            # (B, 12, 5)
        return tf.concat([a, b], axis=-1), yb

    ds = tf.data.Dataset.from_tensor_slices((t, s, y))
    if barajar:
        ds = ds.shuffle(len(t), seed=semilla, reshuffle_each_iteration=True)
    ds = ds.batch(batch).map(extraer, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds, len(t)
