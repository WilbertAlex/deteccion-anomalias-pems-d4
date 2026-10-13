"""
src/g0_linea_base.py  -  G0: linea base estadistica no parametrica (causal).

score[t, s] = |x[t, s] - mu| / sigma, con mu, sigma = media y desv. estandar
(ddof=0) de los `ventana` pasos INMEDIATAMENTE ANTERIORES (t-w .. t-1) del flujo
del sensor s. El paso t no entra en su propia referencia.

Notas de diseno:
  - z-score movil es invariante a transformaciones afines: normalizar o no el
    flujo por pliegue NO cambia el score; se usa el flujo sin normalizar.
  - Si sigma < SIGMA_MIN (ventana practicamente constante) el score es 0:
    decision conservadora, evita falsas alarmas infinitas en noches de flujo 0.
  - Los primeros `ventana` pasos no tienen score (NaN).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SIGMA_MIN = 1e-6


def scores_g0(flow, ventana):
    """flow: (T, S). Retorna scores (T, S) con NaN en los primeros `ventana` pasos."""
    df = pd.DataFrame(flow)
    mu = df.rolling(ventana, min_periods=ventana).mean().shift(1).to_numpy()
    sd = df.rolling(ventana, min_periods=ventana).std(ddof=0).shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(flow - mu) / sd
    z = np.where(sd < SIGMA_MIN, 0.0, z)
    return np.where(np.isnan(mu), np.nan, z)


def _f1(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def evaluar_split(scores, labels, a, b, multiplicadores):
    """
    Evalua por bloque (sensor) en [a, b). Excluye pasos sin etiqueta (-1) o sin
    score (NaN) y bloques sin ambas clases.
    Retorna dict: pr_auc, prevalencia, lift, f1[k], n_bloques (medias sobre bloques).
    """
    pr, prev, f1 = [], [], {k: [] for k in multiplicadores}
    for s in range(scores.shape[1]):
        y, sc = labels[a:b, s], scores[a:b, s]
        ok = (y != -1) & ~np.isnan(sc)
        y, sc = y[ok], sc[ok]
        if len(y) == 0 or y.min() == y.max():
            continue
        pr.append(average_precision_score(y, sc))
        prev.append(y.mean())
        for k in multiplicadores:
            f1[k].append(_f1(y, (sc > k).astype(int)))
    n = len(pr)
    if n == 0:
        return {"n_bloques": 0, "pr_auc": float("nan"), "prevalencia": float("nan"),
                "lift": float("nan"), "f1": {k: float("nan") for k in multiplicadores}}
    return {"n_bloques": n, "pr_auc": float(np.mean(pr)), "prevalencia": float(np.mean(prev)),
            "lift": float(np.mean(pr) / np.mean(prev)),
            "f1": {k: float(np.mean(v)) for k, v in f1.items()}}


def evaluar_test_agregado(scores, labels, a, b, k):
    """AUC-ROC y FPR al umbral k, poolando todos los pasos validos del split."""
    y, sc = labels[a:b].ravel(), scores[a:b].ravel()
    ok = (y != -1) & ~np.isnan(sc)
    y, sc = y[ok], sc[ok]
    pred = sc > k
    fp = int((pred & (y == 0)).sum()); tn = int((~pred & (y == 0)).sum())
    return {"auc_roc_agregado": float(roc_auc_score(y, sc)),
            "fpr": fp / (fp + tn) if (fp + tn) else float("nan")}