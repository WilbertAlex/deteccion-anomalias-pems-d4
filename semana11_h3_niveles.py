"""
Semana 11 - H3 por niveles (revision del asesor, obs. 4). ANALISIS POSTERIOR; la regla se escribe ANTES de calcular.

Regla (fijada aqui, antes de ver el resultado): dos condiciones estan en el mismo NIVEL si su AP medio por sensor difiere en
menos de EPS = 0.01 dentro de cada esquema (agrupamiento por cadenas sobre el orden de AP). Se calcula tau-b de Kendall
(maneja empates) entre los rangos por nivel de A y de B, con IC 95 % por bootstrap de sensores (1000 remuestras, semilla 42)
sobre los sensores comunes. Estabilidad: IC completo >= 0.7; inestable: IC completo < 0.7; inconcluso en otro caso.
Ademas se reporta la posicion de G0 frente al grupo de redes en A y en B.

Entrada: results/prueba/por_bloque.csv (columnas condicion, esquema, split, sensor_idx, pr_auc, evaluable).
Uso:  python semana11_h3_niveles.py
"""
import numpy as np, pandas as pd
from scipy.stats import kendalltau

EPS, B, SEMILLA = 0.01, 1000, 42
df = pd.read_csv("results/prueba/por_bloque.csv")
df = df[(df.split == "prueba") & (df.evaluable == 1)]
def tabla(esq):
    return df[df.esquema == esq].pivot_table(index="sensor_idx", columns="condicion", values="pr_auc", aggfunc="mean")
A, Bt = tabla("A"), tabla("B")
comunes = sorted(set(A.index) & set(Bt.index))
conds = sorted(set(A.columns) & set(Bt.columns))
A, Bt = A.loc[comunes, conds].dropna(), Bt.loc[comunes, conds].dropna()
idx = sorted(set(A.index) & set(Bt.index)); A, Bt = A.loc[idx], Bt.loc[idx]
print(f"sensores comunes: {len(idx)}; condiciones: {conds}")

def niveles(medias):
    """Rango por nivel (0 = mejor). Cadenas: una condicion entra al nivel actual si dista < EPS de la anterior."""
    orden = medias.sort_values(ascending=False)
    nivel, r, prev = {}, 0, None
    for c, v in orden.items():
        if prev is not None and prev - v >= EPS:
            r += 1
        nivel[c] = r; prev = v
    return pd.Series(nivel)

def tau(Ax, Bx):
    na, nb = niveles(Ax.mean()), niveles(Bx.mean())
    return kendalltau(na.loc[conds], nb.loc[conds], variant="b").statistic, na, nb

t0, na, nb = tau(A, Bt)
rng = np.random.default_rng(SEMILLA); taus = []
for _ in range(B):
    i = rng.integers(0, len(idx), len(idx))
    v = tau(A.iloc[i], Bt.iloc[i])[0]
    if np.isfinite(v): taus.append(v)
lo, hi = np.percentile(taus, [2.5, 97.5])
veredicto = "estable" if lo >= 0.7 else ("inestable (H3 apoyada)" if hi < 0.7 else "inconcluso")
print("\nNiveles bajo A:\n", na.sort_values().to_string()); print("\nNiveles bajo B:\n", nb.sort_values().to_string())
print(f"\ntau-b = {t0:.3f}  IC95 [{lo:.3f}, {hi:.3f}]  -> {veredicto}   (EPS={EPS}, n={len(idx)})")
g0 = "G0"
print(f"Posicion de G0: nivel {na[g0]} de {na.max()} bajo A; nivel {nb[g0]} de {nb.max()} bajo B")
