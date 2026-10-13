"""
src/inyeccion_anomalias.py

Modulo reutilizable para el Esquema B: inyeccion controlada de anomalias
sinteticas sobre series reales.

Implementa la Decision 9:
  - Prevalencia objetivo 3% por bloque sensor x pliegue, en cada split.
  - Semilla de inyeccion fija (42) para todas las arquitecturas.
  - Tipos: puntual (1 paso), contextual (P5/P95 del perfil hora x dia del train),
    colectiva (3, 6 o 12 pasos).
  - Distribucion de tipos: 40% puntual, 30% contextual, 30% colectiva.

Las perturbaciones modifican las FEATURES (flow, occupancy, speed), no solo las
etiquetas. Los valores inyectados respetan restricciones fisicas:
  flow >= 0, occupancy en [0, 1], speed >= 0.
"""
import numpy as np


def construir_perfil_hora_dia(flow_train, n_horas=24, n_dias=7):
    """
    Construye un perfil historico (hora x dia) con percentiles 5, 25, 50, 75, 95
    del flujo observado en cada celda del perfil.

    Parametros
    ----------
    flow_train : np.ndarray (T_train, S)
        Flujo crudo del rango de entrenamiento del pliegue.

    Retorna
    -------
    perfil : dict con clave (h, d) y valor dict {p5, p25, p50, p75, p95, n}
    """
    T, S = flow_train.shape
    idx = np.arange(T)
    minutos_desde_inicio = idx * 5
    hora = (minutos_desde_inicio // 60) % 24
    dia_abs = (minutos_desde_inicio // (60 * 24))
    dia_semana = dia_abs % 7

    perfil = {}
    for h in range(n_horas):
        for d in range(n_dias):
            mask = (hora == h) & (dia_semana == d)
            if mask.sum() == 0:
                continue
            valores = flow_train[mask, :].flatten()
            valores = valores[~np.isnan(valores)]
            if len(valores) == 0:
                continue
            perfil[(h, d)] = {
                "p5": float(np.percentile(valores, 5)),
                "p25": float(np.percentile(valores, 25)),
                "p50": float(np.percentile(valores, 50)),
                "p75": float(np.percentile(valores, 75)),
                "p95": float(np.percentile(valores, 95)),
                "n": int(len(valores)),
            }
    return perfil, hora, dia_semana


def _timestamp_a_hora_dia(idx_inicio, idx_fin):
    """Genera arrays de hora y dia_semana para el rango [idx_inicio, idx_fin)."""
    idx = np.arange(idx_inicio, idx_fin)
    minutos = idx * 5
    hora = (minutos // 60) % 24
    dia_abs = minutos // (60 * 24)
    dia_semana = dia_abs % 7
    return hora, dia_semana


def inyectar_anomalias(
    raw_split,
    idx_inicio_split,
    perfil,
    prevalencia_objetivo,
    semilla,
    distribucion_tipos,
    duraciones_colectivas,
    magnitud_sigma,
    factor_reduccion,
):
    """
    Inyecta anomalias en raw_split y devuelve:
      raw_alterado : np.ndarray (T_split, S, 3)
      labels       : np.ndarray (T_split, S) con 1 en anomalias, 0 en normales
    """
    rng = np.random.default_rng(semilla)
    T, S, F = raw_split.shape
    raw_alterado = raw_split.copy()
    labels = np.zeros((T, S), dtype=np.int8)

    hora, dia_semana = _timestamp_a_hora_dia(idx_inicio_split, idx_inicio_split + T)

    n_objetivo_por_sensor = int(T * prevalencia_objetivo)

    tipos = list(distribucion_tipos.keys())
    probs = np.array([distribucion_tipos[t] for t in tipos])
    probs = probs / probs.sum()

    for s in range(S):
        n_inyectados = 0
        intentos = 0
        max_intentos = n_objetivo_por_sensor * 20

        while n_inyectados < n_objetivo_por_sensor and intentos < max_intentos:
            intentos += 1
            tipo = rng.choice(tipos, p=probs)

            if tipo == "puntual":
                t = rng.integers(0, T)
                if labels[t, s] == 1:
                    continue
                t_ini = max(0, t - 288)
                serie = raw_split[t_ini:t, s, 0]
                if len(serie) < 10:
                    continue
                mu = serie.mean()
                sigma = serie.std()
                if sigma <= 0:
                    continue
                k = rng.choice(magnitud_sigma)
                nuevo = mu + k * sigma
                raw_alterado[t, s, 0] = max(0.0, nuevo)
                raw_alterado[t, s, 1] = np.clip(raw_split[t, s, 1] * 1.5, 0, 1)
                raw_alterado[t, s, 2] = max(0.0, raw_split[t, s, 2] * 0.6)
                labels[t, s] = 1
                n_inyectados += 1

            elif tipo == "contextual":
                t = rng.integers(0, T)
                if labels[t, s] == 1:
                    continue
                clave = (int(hora[t]), int(dia_semana[t]))
                if clave not in perfil:
                    continue
                p = perfil[clave]
                lado = rng.choice(["bajo", "alto"])
                if lado == "bajo":
                    nuevo = max(0.0, p["p5"] * rng.uniform(0.3, 0.7))
                else:
                    nuevo = p["p95"] * rng.uniform(1.3, 1.7)
                raw_alterado[t, s, 0] = nuevo
                labels[t, s] = 1
                n_inyectados += 1

            elif tipo == "colectiva":
                dur = int(rng.choice(duraciones_colectivas))
                t_ini = rng.integers(0, max(1, T - dur))
                if labels[t_ini:t_ini + dur, s].any():
                    continue
                factor = rng.uniform(*factor_reduccion)
                for dt in range(dur):
                    t = t_ini + dt
                    prev = raw_split[max(0, t - 1), s, 0]
                    raw_alterado[t, s, 0] = max(0.0, prev * factor)
                    raw_alterado[t, s, 1] = np.clip(raw_split[t, s, 1] * 1.3, 0, 1)
                    raw_alterado[t, s, 2] = max(0.0, raw_split[t, s, 2] * 0.7)
                    labels[t, s] = 1
                n_inyectados += dur

    return raw_alterado, labels