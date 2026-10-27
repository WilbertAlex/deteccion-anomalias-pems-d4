"""
Semana 6 - Verificacion del cargador de datos (src/datos.py). Cada control es un assert.
Comprueba, para A y B y para cada pliegue: forma/tipo, equivalencia exacta con
src/ventanas.py::construir_ventanas, ausencia de etiquetas -1, rangos temporales sin solapamiento
y z-score solo con entrenamiento. No requiere TensorFlow (usa solo la parte NumPy).
Requiere: data/processed/*.npz (A y B), configs/*.json, src/datos.py, src/ventanas.py
Genera:   results/eda/verificacion_datos.json
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")
from datos import Fuente, rangos, VENTANA, N_FEATURES
from ventanas import construir_ventanas

R = []


def chk(nombre, cond, detalle=""):
    R.append({"check": nombre, "ok": bool(cond), "detalle": detalle})
    print(f"  [{'OK ' if cond else 'FAIL'}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    assert cond, f"FALLO: {nombre} {detalle}"


def main():
    conteos = {}
    rng = np.random.default_rng(0)
    for esq in ("A", "B"):
        for pl in (1, 2, 3, "final"):
            f = Fuente(esq, pl)
            print(f"Esquema {esq}, pliegue {pl}")
            chk("X3 float32 y temporales float32", f.X3.dtype == np.float32 and f.temp.dtype == np.float32)
            rg = rangos(pl)
            splits = list(rg)
            tr0, tr1 = rg["train"]
            # z-score solo con entrenamiento: sobre la serie ORIGINAL del train la media normalizada es ~0
            if esq == "A":
                chk("A: media del flujo normalizado en train ~ 0", abs(f.X3[tr0:tr1, :, 0].mean()) < 1e-3,
                    f"{f.X3[tr0:tr1, :, 0].mean():.2e}")
                chk("A: desviacion del flujo normalizado en train ~ 1", abs(f.X3[tr0:tr1, :, 0].std() - 1) < 1e-3)
            for sp in splits:
                a, b = rg[sp]
                t, s, y = f.indices(sp)
                conteos[f"{esq}/pliegue{pl}/{sp}"] = int(len(t))
                chk(f"{sp}: todos los t dentro de [{a},{b}) y t >= {VENTANA - 1}", t.min() >= max(a, VENTANA - 1) and t.max() < b)
                chk(f"{sp}: etiquetas solo 0/1 (sin -1)", set(np.unique(y)) <= {0.0, 1.0})
                # equivalencia exacta con ventanas.py sobre una muestra de pares (t, s)
                # equivalencia exacta con ventanas.py: 5 instantes t al azar, TODOS sus sensores validos
                feats = np.concatenate([f.X3, np.broadcast_to(f.temp[:, None, :], (f.T, f.S, 5))], axis=2)
                ts = rng.choice(np.unique(t), size=5, replace=False)
                n_cmp = 0
                for tv in ts:
                    pos = np.where(t == tv)[0]
                    W = f.ventanas_numpy(t[pos], s[pos])
                    assert W.shape[1:] == (VENTANA, N_FEATURES) and W.dtype == np.float32
                    X, meta = construir_ventanas(feats, int(tv), int(tv) + 1)
                    ref = {int(m): X[i] for i, m in enumerate(meta[:, 0])}
                    for w, sv in zip(W, s[pos]):
                        assert np.array_equal(w, ref[int(sv)]), "ventana distinta a src/ventanas.py"
                        n_cmp += 1
                chk(f"{sp}: forma (n,{VENTANA},{N_FEATURES}) float32 y ventanas identicas a src/ventanas.py", n_cmp > 0,
                    f"{n_cmp} ventanas comparadas en 5 instantes")
            # train y val/test no comparten instantes
            for sp in splits[1:]:
                t1 = f.indices("train")[0]; t2 = f.indices(sp)[0]
                chk(f"train y {sp} sin instantes comunes", t1.max() < t2.min())
    # recuentos conocidos (revision del 13 oct.): pliegue 3, train, esquema B = 291 * (10194 - 11)
    chk("B pliegue 3 train: 291 * (10194 - 11) = 2 963 253 ventanas", conteos["B/pliegue3/train"] == 291 * (10194 - 11),
        str(conteos["B/pliegue3/train"]))
    ok = sum(r["ok"] for r in R)
    print(f"\nResultado: {ok}/{len(R)} controles OK")
    Path("results/eda").mkdir(parents=True, exist_ok=True)
    Path("results/eda/verificacion_datos.json").write_text(
        json.dumps({"n_ok": ok, "n_total": len(R), "conteos_ventanas": conteos, "checks": R}, indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()
