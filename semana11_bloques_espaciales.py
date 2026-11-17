"""
Semana 11 - Bootstrap por bloques espaciales (revision del asesor, obs. 5). POSTERIOR Y EXPLORATORIO.

Los 291 sensores no son independientes (autopistas vecinas). Se agrupan en bloques por cercania en la red vial y se
remuestrean BLOQUES completos (bootstrap por conglomerados) para obtener el IC 95 % de la diferencia media de AP por sensor
(condicion - G0, Esquema B, prueba), que se compara con el IC ingenuo por sensor.
Entradas:
  results/prueba/por_bloque.csv   (condicion, esquema, split, sensor_idx, pr_auc, evaluable)
  distance.csv de PEMS04 (columnas from,to,cost; la de ASTGCN/PeMS), via --distancia
  Ids de sensor: --diagnostico imprime las llaves de data/processed/pems04_semana2.npz para saber como se mapean los 291
  indices a los 307 ids originales. Si el npz trae 'sensor_ids' o similar, pasa --claves <nombre>; si no, pasa --indices
  <archivo.txt> con los ids originales (uno por linea) de los 291 sensores en el orden del npz.
Uso:
  python semana11_bloques_espaciales.py --diagnostico
  python semana11_bloques_espaciales.py --distancia ruta\\distance.csv [--claves sensor_ids | --indices ids.txt] [--bloques 30]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd


def diagnostico():
    z = np.load("data/processed/pems04_semana2.npz", allow_pickle=True)
    for k in z.files:
        print(k, z[k].shape, z[k].dtype)
    for f in Path("data/processed").glob("*"):
        print("archivo:", f.name)
    for f in list(Path(".").glob("**/*distance*"))[:5]:
        print("posible distance:", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostico", action="store_true"); ap.add_argument("--distancia")
    ap.add_argument("--claves"); ap.add_argument("--indices"); ap.add_argument("--bloques", type=int, default=30)
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()
    if a.diagnostico: return diagnostico()
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    z = np.load("data/processed/pems04_semana2.npz", allow_pickle=True)
    if a.claves: ids = np.asarray(z[a.claves]).astype(int)
    elif a.indices: ids = np.loadtxt(a.indices, dtype=int)
    else: sys.exit("Indica --claves o --indices (ver --diagnostico)")
    d = pd.read_csv(a.distancia); d.columns = [c.lower() for c in d.columns]
    n = int(max(d["from"].max(), d["to"].max())) + 1
    M = csr_matrix((d["cost"].astype(float), (d["from"].astype(int), d["to"].astype(int))), shape=(n, n))
    D = shortest_path(M, directed=False)
    fin = D[np.isfinite(D)]; D[~np.isfinite(D)] = fin.max() * 1.5
    Ds = D[np.ix_(ids, ids)]; Ds = (Ds + Ds.T) / 2; np.fill_diagonal(Ds, 0)
    blq = fcluster(linkage(squareform(Ds, checks=False), "average"), t=a.bloques, criterion="maxclust")
    print(f"{len(ids)} sensores en {len(set(blq))} bloques; tamano medio {len(ids)/len(set(blq)):.1f}")
    df = pd.read_csv("results/prueba/por_bloque.csv")
    df = df[(df.esquema == "B") & (df.split == "prueba") & (df.evaluable == 1)]
    W = df.pivot_table(index="sensor_idx", columns="condicion", values="pr_auc")
    rng = np.random.default_rng(a.semilla); filas = []
    bloque_de = pd.Series(blq, index=range(len(ids)))
    for c in [x for x in W.columns if x != "G0"]:
        dif = (W[c] - W["G0"]).dropna()
        grp = bloque_de.loc[dif.index]
        agrup = [dif[grp == g].to_numpy() for g in sorted(grp.unique())]
        ing = [dif.sample(len(dif), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(a.B)]
        esp = []
        for _ in range(a.B):
            sel = rng.integers(0, len(agrup), len(agrup))
            esp.append(np.concatenate([agrup[i] for i in sel]).mean())
        filas.append({"condicion": c, "n": len(dif), "dif_media": dif.mean(),
                      "IC_ingenuo": f"[{np.percentile(ing,2.5):.3f}, {np.percentile(ing,97.5):.3f}]",
                      "IC_bloques": f"[{np.percentile(esp,2.5):.3f}, {np.percentile(esp,97.5):.3f}]"})
    out = pd.DataFrame(filas); print(out.to_string(index=False))
    Path("results/eda").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/eda/bootstrap_bloques_espaciales.csv", index=False)


if __name__ == "__main__":
    main()
