"""
G7 - Paso 2: toma los datos YA PROCESADOS del proyecto (los mismos que usan G0-G6) y los
verifica contra las cifras del PPI antes de entrenar.

Uso (desde D:\\ppi\\SI-DMSeq, con .venv-g7 activo):
    python g7_preparar.py --proyecto D:\\ppi\\deteccion-anomalias-pems-d4

Lee de <proyecto>\\data\\processed:
  pems04_semana2.npz          serie base de los 291 sensores (data, sensores_retenidos)
  labels_esquema_A.npz        etiquetas del Esquema A
  esquema_B_pliegue{1,2,3}.npz  serie con inyecciones y etiquetas de cada pliegue (Decision 10)
No usa los archivos *_v1_superado (Esquema B sustituido) ni esquema_B_final (prueba final).

Verificaciones (si alguna falla, no guarda nada):
  V1  Serie base: 291 sensores retenidos de 307 (16 excluidos)                   (sec. 2.2)
  V2  Particion: prueba [13593, 16992); pliegues 0 | 3398 | 6796 | 10194 | 13593;
      ventanas de entrenamiento del pliegue 3 = 2 963 253                         (Anexo C, sem. 3 y 8)
  V3  Media del flujo en el entrenamiento de cada pliegue: 209,97 / 210,40 / 211,83 (Anexo C, sem. 3)
  V4  Esquema A: 4 873 anomalos de 4 860 864 instantes validos                  (Anexo C, sem. 4)
  V5  Esquema A: 256 de 873 bloques con ambas clases en validacion              (Anexo B)
  V6  Esquema B: prevalencia entre 2,9 % y 3,2 % en entrenamiento y validacion
      de cada pliegue, y 291 de 291 bloques con ambas clases en validacion       (Anexo C, sem. 5)

Salida: g7_datos.npz y g7_verificaciones.json
"""
import argparse
import json
import os
import sys

import numpy as np

LIM = [0, 3398, 6796, 10194, 13593]
FIN, VENTANA = 16992, 12


def cargar(ruta, *claves):
    if not os.path.exists(ruta):
        sys.exit(f'No existe {ruta}')
    with np.load(ruta) as z:
        return [z[k] for k in claves]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proyecto', required=True)
    ap.add_argument('--forzar', action='store_true', help='guardar aunque falle una verificacion (solo si se indica)')
    a = ap.parse_args()
    proc = os.path.join(a.proyecto, 'data', 'processed')

    X, ret = cargar(os.path.join(proc, 'pems04_semana2.npz'), 'data', 'sensores_retenidos')
    (yA,) = cargar(os.path.join(proc, 'labels_esquema_A.npz'), 'labels')
    XB, yB = {}, {}
    for k in (1, 2, 3):
        XB[k], yB[k] = cargar(os.path.join(proc, f'esquema_B_pliegue{k}.npz'), 'data', 'labels')
    S = X.shape[1]
    ver = {}

    ver['V1_sensores'] = dict(forma=list(X.shape), retenidos=int(len(ret)), unicos=int(len(np.unique(ret))),
                              ok=bool(X.shape == (FIN, 291, 3) and len(np.unique(ret)) == 291
                                      and ret.min() >= 0 and ret.max() < 307))

    v_tr3 = (LIM[3] - (VENTANA - 1)) * S
    ver['V2_particion'] = dict(limites=LIM, prueba=[LIM[4], FIN], ventanas_train_p3=int(v_tr3),
                               ok=bool(v_tr3 == 2963253))

    medias = [round(float(X[:LIM[k], :, 0].mean()), 2) for k in (1, 2, 3)]
    ver['V3_medias_flujo'] = dict(pliegues=medias, esperado=[209.97, 210.40, 211.83],
                                  ok=bool(all(abs(m - e) < 0.011 for m, e in zip(medias, [209.97, 210.40, 211.83]))))

    ver['V4_esquemaA'] = dict(valores=sorted(int(v) for v in np.unique(yA)), anomalos=int((yA == 1).sum()),
                              validos=int((yA >= 0).sum()),
                              ok=bool((yA == 1).sum() == 4873 and (yA >= 0).sum() == 4860864))

    nb = 0
    for k in (1, 2, 3):
        yv = yA[LIM[k]:LIM[k + 1]]
        nb += int(((yv == 1).any(0) & (yv == 0).any(0)).sum())
    ver['V5_bloques_A'] = dict(con_ambas_clases=nb, total=3 * S, ok=bool(nb == 256))

    detB, okB = {}, True
    for k in (1, 2, 3):
        tr, va = yB[k][:LIM[k]], yB[k][LIM[k]:LIM[k + 1]]
        p_tr = float((tr == 1).sum() / max(1, (tr >= 0).sum()))
        p_va = float((va == 1).sum() / max(1, (va >= 0).sum()))
        bloques = int(((va == 1).any(0) & (va == 0).any(0)).sum())
        cambios = int((XB[k] != X).any(2).sum())
        detB[f'pliegue{k}'] = dict(prev_train=round(100 * p_tr, 3), prev_val=round(100 * p_va, 3),
                                   bloques_val=bloques, instantes_alterados=cambios)
        okB &= 0.029 <= p_tr <= 0.032 and 0.029 <= p_va <= 0.032 and bloques == S
    ver['V6_esquemaB'] = dict(**detB, ok=bool(okB))

    todo_ok = all(v['ok'] for v in ver.values())
    for k, v in ver.items():
        print(f'{"OK   " if v["ok"] else "FALLA"} {k}: { {kk: vv for kk, vv in v.items() if kk != "ok"} }')
    with open('g7_verificaciones.json', 'w', encoding='utf-8') as f:
        json.dump(ver, f, indent=1, ensure_ascii=False)
    if not todo_ok and not a.forzar:
        sys.exit('\nHay verificaciones que no coinciden con el PPI. No se guardo g7_datos.npz.\n'
                 'Comparte esta salida y g7_verificaciones.json antes de entrenar.')

    np.savez('g7_datos.npz', X=X.astype(np.float32), validos=ret, yA=yA,
             XB1=XB[1].astype(np.float32), XB2=XB[2].astype(np.float32), XB3=XB[3].astype(np.float32),
             yB1=yB[1], yB2=yB[2], yB3=yB[3], limites=np.array(LIM), fin=FIN)
    print(f'\nDatos listos: g7_datos.npz ({S} sensores, Esquemas A y B). Verificaciones en g7_verificaciones.json')


if __name__ == '__main__':
    main()
