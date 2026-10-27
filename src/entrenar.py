"""
src/entrenar.py - Utilidades comunes de entrenamiento para G1-G6 (TensorFlow/Keras).

IMPORTANTE: los hiperparametros por defecto de `construir_modelo_recurrente` son SOLO para medir
costos y probar el pipeline. NO son la configuracion inicial del Cuadro 6 del PPI; esa se debe
pasar explicitamente cuando se lance la busqueda real.

Funciones:
  fijar_semillas(seed)             semillas de Python/NumPy/TensorFlow
  dispositivos()                   lista de GPU/CPU visibles (para registrar el entorno)
  construir_modelo_recurrente(...) LSTM o GRU -> Dense(1, sigmoid), entrada (12, 8)
  entrenar(...)                    fit con parada temprana por PR-AUC de validacion (metrica de Keras,
                                   agregada sobre todos los sensores = PROXY del PR-AUC por bloque)
  evaluar_por_bloque(...)          PR-AUC por sensor en el mismo formato que results/eda/g0_por_bloque.csv
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np


def fijar_semillas(seed):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)   # fija Python, NumPy y TensorFlow


def dispositivos():
    import tensorflow as tf
    return {"gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
            "cpu": [d.name for d in tf.config.list_physical_devices("CPU")],
            "tensorflow": tf.__version__}


def construir_modelo_recurrente(tipo="lstm", unidades=64, capas=1, dropout=0.0, lr=1e-3,
                                ventana=12, n_features=8):
    import tensorflow as tf
    cap = {"lstm": tf.keras.layers.LSTM, "gru": tf.keras.layers.GRU}[tipo]
    x = inp = tf.keras.Input(shape=(ventana, n_features), dtype="float32")
    for i in range(capas):
        x = cap(unidades, return_sequences=(i < capas - 1), dropout=dropout)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    return m


def entrenar(modelo, ds_train, ds_val, epocas, paciencia=5, pasos_por_epoca=None, verbose=2):
    """Parada temprana por PR-AUC de validacion (mayor es mejor); restaura los mejores pesos."""
    import tensorflow as tf
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=paciencia,
                                           restore_best_weights=True)]
    t0 = time.perf_counter()
    h = modelo.fit(ds_train, validation_data=ds_val, epochs=epocas, steps_per_epoch=pasos_por_epoca,
                   callbacks=cb, verbose=verbose, shuffle=False)  # el barajado lo hace tf.data
    return {"segundos": time.perf_counter() - t0,
            "epocas_ejecutadas": len(h.history["loss"]),
            "mejor_val_pr_auc": float(max(h.history["val_pr_auc"]))}


def scores(modelo, fuente, split, batch=1024):
    """Scores (T_rango, S) del modelo sobre el split (NaN donde no hay ventana o etiqueta)."""
    from datos import dataset
    a, b = fuente.rangos[split]
    t, s, _ = fuente.indices(split)
    ds, n = dataset(fuente, split, batch=batch, barajar=False)
    p = modelo.predict(ds, verbose=0).reshape(-1)
    assert len(p) == n
    M = np.full((b - a, fuente.S), np.nan, dtype=np.float32)
    M[t - a, s] = p
    return M, a, b


def evaluar_por_bloque(modelo, fuente, split, condicion, esquema, nombre_split):
    """Filas (dict) PR-AUC por sensor, mismo formato que g0_por_bloque.csv."""
    from sklearn.metrics import average_precision_score
    M, a, b = scores(modelo, fuente, split)
    filas = []
    for s in range(fuente.S):
        y = fuente.labels[a:b, s]
        sc = M[:, s]
        ok = (y != -1) & ~np.isnan(sc)
        y, sc = y[ok], sc[ok]
        evaluable = len(y) > 0 and y.min() != y.max()
        pr = float(average_precision_score(y, sc)) if evaluable else float("nan")
        filas.append({"condicion": condicion, "esquema": esquema, "split": nombre_split, "sensor_idx": s,
                      "pr_auc": "" if np.isnan(pr) else f"{pr:.10f}", "n_validos": int(len(y)),
                      "n_positivos": int((y == 1).sum()), "evaluable": int(evaluable)})
    return filas
