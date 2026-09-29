"""
src/ventanas.py

Modulo reutilizable para construir ventanas temporales multivariadas.
Sera importado por los scripts de entrenamiento de las Semanas 7 en adelante
(G0-G7), para no duplicar esta logica 8 veces.

Unidad de analisis (Seccion 2.2 del protocolo): para un sensor s y un
instante t, la ventana de entrada contiene los 12 pasos entre t-11 y t
inclusive. La etiqueta (cuando exista, Semana 4-5) corresponde al instante
final t.
"""
import numpy as np


def construir_ventanas(features, idx_start, idx_end, window=12):
    """
    Construye ventanas para todos los sensores, con instante final t
    recorriendo el rango [idx_start, idx_end) (medio-abierto, igual que
    la convencion usada en particion_temporal.json).

    Parametros
    ----------
    features : np.ndarray, forma (n_steps, n_sensores, n_variables)
        Arreglo ya normalizado (variables crudas) y con variables
        temporales concatenadas.
    idx_start, idx_end : int
        Rango de instantes t a generar (ej. train_idx o val_idx del pliegue).
    window : int
        Longitud de la ventana (12 por defecto, seccion 2.2).

    Retorna
    -------
    X : np.ndarray, forma (n_ventanas, window, n_variables)
        n_ventanas = n_sensores * max(0, idx_end - max(idx_start, window-1))
    meta : np.ndarray, forma (n_ventanas, 2)
        Columna 0: sensor_id (indice dentro del arreglo filtrado, 0-290)
        Columna 1: indice temporal t (instante final de la ventana)
    """
    n_steps, n_sensores, n_vars = features.shape
    t_min = max(idx_start, window - 1)  # no se puede formar ventana si t < window-1
    t_values = np.arange(t_min, idx_end)

    if len(t_values) == 0:
        return (np.empty((0, window, n_vars)), np.empty((0, 2), dtype=int))

    n_ventanas = len(t_values) * n_sensores
    X = np.empty((n_ventanas, window, n_vars), dtype=features.dtype)
    meta = np.empty((n_ventanas, 2), dtype=int)

    idx = 0
    for t in t_values:
        bloque = features[t - window + 1: t + 1, :, :]           # (window, n_sensores, n_vars)
        bloque = np.transpose(bloque, (1, 0, 2))                  # (n_sensores, window, n_vars)
        X[idx: idx + n_sensores] = bloque
        meta[idx: idx + n_sensores, 0] = np.arange(n_sensores)
        meta[idx: idx + n_sensores, 1] = t
        idx += n_sensores

    return X, meta
