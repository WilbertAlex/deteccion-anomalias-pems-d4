"""
G7 - Paso 1: inventario de datos del proyecto (solo lectura, no modifica nada).

Uso (desde D:\\ppi\\SI-DMSeq, con .venv-g7 activo):
    python g7_inspeccionar.py --proyecto D:\\ppi\\deteccion-anomalias-pems-d4

Busca todos los .npz/.npy del proyecto (fuera de entornos virtuales), lista sus claves con
forma y tipo, identifica el archivo crudo de PeMS-D4 (forma 16992 x 307 x 3), calcula su
SHA-256 y lo compara con el registrado en el PPI (95a3c9b7...c999a). Guarda todo en
g7_inventario.txt. Ese archivo es el que se comparte para conectar el Esquema B.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HASH_INICIO, HASH_FIN = '95a3c9b7', 'c999a'


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for bloque in iter(lambda: f.read(1 << 20), b''):
            h.update(bloque)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proyecto', required=True, help='carpeta raiz del repositorio del proyecto')
    a = ap.parse_args()
    if not os.path.isdir(a.proyecto):
        sys.exit(f'No existe la carpeta {a.proyecto}')

    lineas, crudos = [], []
    for raiz, dirs, archivos in os.walk(a.proyecto):
        dirs[:] = [d for d in dirs if 'venv' not in d.lower() and not d.startswith('.git')]
        for fn in sorted(archivos):
            if not fn.lower().endswith(('.npz', '.npy')):
                continue
            ruta = os.path.join(raiz, fn)
            rel = os.path.relpath(ruta, a.proyecto)
            mb = os.path.getsize(ruta) / 1e6
            try:
                if fn.lower().endswith('.npz'):
                    with np.load(ruta, allow_pickle=False) as z:
                        claves = {k: (z[k].shape, str(z[k].dtype)) for k in z.files}
                else:
                    arr = np.load(ruta, mmap_mode='r', allow_pickle=False)
                    claves = {'(npy)': (arr.shape, str(arr.dtype))}
            except Exception as e:  # archivo con objetos o danado
                claves = {'ERROR': str(e)[:80]}
            lineas.append(f'{rel}  [{mb:.1f} MB]')
            for k, v in claves.items():
                lineas.append(f'    {k}: {v}')
                if isinstance(v, tuple) and tuple(v[0]) == (16992, 307, 3):
                    crudos.append(ruta)

    lineas.append('')
    if crudos:
        for r in crudos:
            h = sha256(r)
            ok = h.startswith(HASH_INICIO) and h.endswith(HASH_FIN)
            lineas.append(f'CRUDO PeMS-D4: {r}')
            lineas.append(f'    SHA-256 {h}  -> {"COINCIDE con el PPI" if ok else "NO COINCIDE con el PPI"}')
    else:
        lineas.append('CRUDO PeMS-D4: no se encontro ningun arreglo de forma (16992, 307, 3)')

    texto = '\n'.join(lineas)
    print(texto)
    with open('g7_inventario.txt', 'w', encoding='utf-8') as f:
        f.write(texto + '\n')
    print('\nInventario guardado en g7_inventario.txt')


if __name__ == '__main__':
    main()
