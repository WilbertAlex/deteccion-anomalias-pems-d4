"""
G7 (SI-DMSeq) - reentrenamiento final sobre el bloque 80 % (PPI 2.7.1, paso 3), igual que G1-G6 en
semana10_reentrenamiento.py, pero con la configuracion por defecto de los autores (sin busqueda).

Uso (desde D:\\ppi\\SI-DMSeq, con .venv-g7 activo):
    python g7_final.py --proyecto D:\\ppi\\deteccion-anomalias-pems-d4 --esquema A
    python g7_final.py --proyecto D:\\ppi\\deteccion-anomalias-pems-d4 --esquema B
    python g7_final.py --proyecto ... --esquema A --epocas 1      # prueba corta (se guarda en resultados_g7_final_PRUEBA)

Reglas (mismas que G1-G6 y que la validacion de G7):
  * Entrenamiento: pasos [0, 13593) (bloque 80 %), todas las ventanas.
  * Esquema A: serie original (pems04_semana2.npz). Esquema B: esquema_B_final.npz.
  * Normalizacion z-score por variable con media y desviacion del bloque 80 % de la serie ORIGINAL (Decision 14).
  * Configuracion por defecto de los autores: ventana 24, embedding 64, 2 capas, dropout 0.2, Adam lr 0.01,
    MSE suma, lote 64, 20 epocas, rho 0.25; semillas fijas del codigo original (determinista: 1 corrida por esquema).
  * 873 canales (flujo, ocupacion y velocidad de los 291 sensores).
  * Score (sec. 2.6.1): error de reconstruccion del ultimo paso de la ventana, suma por sensor de sus 3 canales.
  * Se puntua SOLO el bloque del umbral (ultimo bloque de validacion del pliegue 3, pasos [10194, 13593)).
    La prueba final NO se toca aqui.

Salidas en resultados_g7_final/:
  g7_final.jsonl               una linea por esquema
  g7_<esq>_final.pth           pesos del modelo
  g7_<esq>_final_umbral.npz    score (3399 x 291), etiquetas y limites del bloque del umbral
"""
import argparse
import json
import os
import platform
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn, optim

sys.path.insert(0, os.getcwd())
try:
    from src.model import MultitaskSequenceModel
    from src.train import train
    from g7_entrenar import FastTrafficDataset, ventanas_eval, scores, DEF
except ImportError as e:
    sys.exit(f'Falta codigo ({e}). Ejecuta desde D:\\ppi\\SI-DMSeq, donde estan src\\ y g7_entrenar.py')

FIN_80, UMB_INI, UMB_FIN = 13593, 10194, 13593


def cargar(ruta, *claves):
    if not os.path.exists(ruta):
        sys.exit(f'No existe {ruta}')
    with np.load(ruta) as z:
        return [z[k] for k in claves]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proyecto', required=True)
    ap.add_argument('--esquema', choices=['A', 'B'], required=True)
    ap.add_argument('--epocas', type=int, default=DEF['epochs'])
    a = ap.parse_args()
    prueba = a.epocas != DEF['epochs']
    sal = 'resultados_g7_final_PRUEBA' if prueba else 'resultados_g7_final'
    if prueba:
        shutil.rmtree(sal, ignore_errors=True)
    os.makedirs(sal, exist_ok=True)
    reg_path = os.path.join(sal, 'g7_final.jsonl')
    if not prueba and os.path.exists(reg_path):
        hechos = [json.loads(l)['esquema'] for l in open(reg_path, encoding='utf-8') if l.strip()]
        if a.esquema in hechos:
            sys.exit(f'G7 {a.esquema} final ya estaba en {reg_path}; no se repite.')
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    proc = os.path.join(a.proyecto, 'data', 'processed')
    (X0,) = cargar(os.path.join(proc, 'pems04_semana2.npz'), 'data')
    if a.esquema == 'A':
        (y,) = cargar(os.path.join(proc, 'labels_esquema_A.npz'), 'labels')
        X = X0
    else:
        X, y = cargar(os.path.join(proc, 'esquema_B_final.npz'), 'data', 'labels')
    assert X.shape == X0.shape and X.shape[1] == 291 and X.shape[2] == 3, X.shape
    T_, S, C = X.shape
    L = DEF['seq_len']

    mu = X0[:FIN_80].reshape(-1, C).mean(0)                         # Decision 14: serie original
    sd = X0[:FIN_80].reshape(-1, C).std(0)
    Z = ((X - mu) / sd).astype(np.float32)
    M = np.ascontiguousarray(Z.transpose(0, 2, 1).reshape(T_, C * S))
    tr = FastTrafficDataset(M[:FIN_80], L, DEF['y_len'], C * S, 'train', DEF['rho'])
    w_umb = ventanas_eval(M[UMB_INI - L + 1:UMB_FIN], L)
    y_umb = y[UMB_INI:UMB_FIN]
    assert len(w_umb) == len(y_umb) == UMB_FIN - UMB_INI

    print(f'=== G7 {a.esquema} final: entrenamiento [0, {FIN_80}), {C * S} canales, {len(tr)} ventanas, '
          f'{a.epocas} epocas; umbral [{UMB_INI}, {UMB_FIN}) ===', flush=True)
    torch.manual_seed(1)
    model = MultitaskSequenceModel(feature_size=C * S, embedding_size=DEF['embedding_size'],
                                   num_layers=DEF['num_layers'], dropout=DEF['dropout'])
    opt = optim.Adam(model.parameters(), lr=DEF['lr'])
    t0 = time.perf_counter()
    hist = train(model, tr, opt, nn.MSELoss(reduction='sum'), None, batch_size=DEF['batch_size'],
                 epochs=a.epocas, verbose=False, device=torch.device('cpu'))
    seg = time.perf_counter() - t0
    perd = [round(float(v), 2) for v in hist['train']]
    assert all(np.isfinite(perd)), 'perdida no finita'
    sc = scores(model, w_umb, S, C)
    assert sc.shape == (UMB_FIN - UMB_INI, S) and np.isfinite(sc).all()

    torch.save(model.state_dict(), os.path.join(sal, f'g7_{a.esquema}_final.pth'))
    np.savez_compressed(os.path.join(sal, f'g7_{a.esquema}_final_umbral.npz'), score=sc.astype(np.float32),
                        labels=y_umb, inicio=UMB_INI, fin=UMB_FIN)
    reg = dict(arq='si_dmseq', esquema=a.esquema, epocas=a.epocas, semilla='autores', canales=C * S,
               ventanas_train=len(tr), segundos=round(seg, 1), perdida_train=perd,
               bloque_entrenamiento=[0, FIN_80], bloque_umbral=[UMB_INI, UMB_FIN],
               maquina=platform.node(), torch=torch.__version__)
    with open(reg_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(reg, ensure_ascii=False) + '\n')
    print(f'FINAL G7 {a.esquema}: {a.epocas} ep, {seg / 60:.1f} min, {len(tr)} ventanas, '
          f'perdida {perd[0]} -> {perd[-1]}', flush=True)
    if prueba:
        shutil.rmtree(sal)
        print('Prueba corta de G7 final: OK (resultados de prueba borrados)')


if __name__ == '__main__':
    main()
