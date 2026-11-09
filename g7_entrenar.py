"""
G7 (SI-DMSeq) - Paso 3: entrenamiento y evaluacion en los pliegues de validacion de PeMS-D4.

Uso (desde D:\\ppi\\SI-DMSeq, con .venv-g7 activo):
    python g7_entrenar.py --esquema A                       # 3 pliegues, configuracion de los autores
    python g7_entrenar.py --esquema A --pliegues 1          # solo un pliegue (prueba corta)
    python g7_entrenar.py --esquema A --epocas 2 --pliegues 1   # muestra reducida (minutos)

Protocolo (PPI sec. 2.6, 2.6.1, 2.7.1 y 3.1):
  * Modelo, funcion train() y configuracion POR DEFECTO de los autores, sin busqueda de
    hiperparametros: ventana 24, embedding 64, 2 capas, dropout 0.2, Adam lr 0.01, MSE suma,
    lote 64, 20 epocas, rho 0.25. El codigo de src/ no se modifica.
  * Semillas: las que fijan los autores en su codigo (0 datos, 1 modelo, 4 entrenamiento):
    la configuracion publicada es determinista -> 1 corrida por esquema y pliegue (6 en total).
    Con --semillas 42 43 44 se cambia solo la semilla de inicializacion del modelo.
  * Entrada: la serie de cada sensor es un canal del modelo (como las zonas en el original).
    --canales todos (defecto): flujo, ocupacion y velocidad de los 291 sensores (873 canales).
    --canales flujo: solo flujo (291 canales).
    Las 5 variables temporales no se incluyen: el modelo trata el valor 0 como dato faltante,
    y sin/cos y la hora pico valen 0 con frecuencia.
  * Normalizacion z-score por variable con la media y desviacion del entrenamiento del pliegue,
    calculadas siempre sobre la serie ORIGINAL (sin inyecciones), en ambos esquemas (Decision 14).
  * Esquema B: serie y etiquetas del arreglo inyectado de cada pliegue (esquema_B_pliegue<k>).
  * Score puntual (sec. 2.6.1): error de reconstruccion local del sensor s en el instante t
    (ultima posicion de la ventana), suma de los cuadrados de sus canales; sin agregacion por
    red ni carta de control.
  * Metrica: PR-AUC por bloque sensor x pliegue (definicion de sklearn average_precision),
    solo bloques con ambas clases. La prueba final NO se toca aqui.

Salidas en resultados_g7/:
  resultados_g7.jsonl                       una linea por corrida (como la busqueda de G1-G6)
  g7_<esq>_p<k>_s<semilla>_<canales>.npz    score, etiquetas y PR-AUC por bloque (para Friedman)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

sys.path.insert(0, os.getcwd())
try:
    from src.data_utils import TrafficDataset
    from src.model import MultitaskSequenceModel
    from src.train import train
except ImportError as e:
    sys.exit(f'No se encuentra el codigo de los autores ({e}). Ejecuta desde la raiz de SI-DMSeq.')

DEF = dict(seq_len=24, y_len=1, embedding_size=64, num_layers=2, dropout=0.2,
           lr=0.01, batch_size=64, epochs=20, rho=0.25)
SALIDA = 'resultados_g7'


class FastTrafficDataset(TrafficDataset):
    """Identico a TrafficDataset de los autores (verificado en smoke_g7.py), construido con unfold."""

    def __init__(self, matriz, seq_len, y_len, feature_size, mode='train', rho=0.25):
        self.rho = rho if mode == 'train' else .0
        torch.manual_seed(0)
        comp = torch.as_tensor(matriz)
        incomp = F.dropout(comp, p=self.rho) * (1 - self.rho)
        n = comp.shape[0] - seq_len - y_len + 1
        self.seqs = incomp[:n + seq_len - 1].unfold(0, seq_len, 1).permute(0, 2, 1)[:n].contiguous()
        self.targets = incomp[seq_len:].unfold(0, y_len, 1).permute(0, 2, 1)[:n].contiguous()
        del comp, incomp
        self.seq_masks = (self.seqs == 0)
        mean = (self.seqs.sum(dim=2) / (~self.seq_masks).sum(dim=2)).unsqueeze(2)
        self.seqs += self.seq_masks * mean
        self.seq_len = seq_len
        self.target_masks = (self.targets == 0)


def ventanas_eval(matriz, seq_len):
    """Ventanas que terminan en cada instante de la matriz desde seq_len-1; mismo relleno de ceros."""
    m = torch.as_tensor(matriz)
    w = m.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()
    mask = (w == 0)
    w += mask * (w.sum(dim=2) / (~mask).sum(dim=2)).unsqueeze(2)
    return w


@torch.no_grad()
def scores(model, w, n_sens, n_can, lote=128):
    model.eval()
    out = []
    for i in range(0, len(w), lote):
        x = w[i:i + lote]
        recon, _ = model(x)
        e = (recon[:, -1, :] - x[:, -1, :]) ** 2                    # (lote, canales)
        out.append(e.reshape(len(x), n_can, n_sens).sum(1).numpy())  # suma por sensor
    return np.concatenate(out, 0)                                   # (instantes, sensores)


def average_precision(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    orden = np.argsort(-s, kind='mergesort')
    y, s = y[orden], s[orden]
    corte = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[corte]
    prec = tp / (corte + 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--esquema', choices=['A', 'B'], required=True)
    ap.add_argument('--pliegues', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--canales', choices=['todos', 'flujo'], default='todos')
    ap.add_argument('--semillas', type=int, nargs='+', default=None, help='por defecto, las de los autores')
    ap.add_argument('--epocas', type=int, default=DEF['epochs'])
    ap.add_argument('--datos', default='g7_datos.npz')
    a = ap.parse_args()

    if not os.path.exists(a.datos):
        sys.exit(f'No existe {a.datos}. Ejecuta antes g7_preparar.py')
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(SALIDA, exist_ok=True)

    d = np.load(a.datos)
    LIM = d['limites']
    nc = 1 if a.canales == 'flujo' else 3
    X0 = d['X'][:, :, :nc]                                          # serie original (T, S, C)
    T_, S, C = X0.shape
    semillas = a.semillas or [None]
    L = DEF['seq_len']

    for k in a.pliegues:
        tr_fin, va_ini, va_fin = int(LIM[k]), int(LIM[k]), int(LIM[k + 1])
        mu = X0[:tr_fin].reshape(-1, C).mean(0)                    # Decision 14: desde la serie original
        sd = X0[:tr_fin].reshape(-1, C).std(0)
        if a.esquema == 'A':
            X, y_all = X0, d['yA']
        else:
            X, y_all = d[f'XB{k}'][:, :, :nc], d[f'yB{k}']
        Z = ((X - mu) / sd).astype(np.float32)
        # canales ordenados variable-mayor: [flujo s0..sN, ocup s0..sN, vel s0..sN]
        M = np.ascontiguousarray(Z.transpose(0, 2, 1).reshape(T_, C * S))
        tr = FastTrafficDataset(M[:tr_fin], L, DEF['y_len'], C * S, 'train', DEF['rho'])
        va = FastTrafficDataset(M[va_ini - L + 1:va_fin], L, DEF['y_len'], C * S, 'val', DEF['rho'])
        w_eval = ventanas_eval(M[va_ini - L + 1:va_fin], L)        # una ventana por instante de val
        y = y_all[va_ini:va_fin]
        assert len(w_eval) == len(y) == va_fin - va_ini

        for sem in semillas:
            etiqueta = f'g7_{a.esquema}_p{k}_s{"aut" if sem is None else sem}_{a.canales}'
            print(f'\n=== {etiqueta}: entrenamiento [0, {tr_fin}) -> validacion [{va_ini}, {va_fin}), '
                  f'{C * S} canales, {len(tr)} ventanas, {a.epocas} epocas ===', flush=True)
            torch.manual_seed(1 if sem is None else sem)
            model = MultitaskSequenceModel(feature_size=C * S, embedding_size=DEF['embedding_size'],
                                           num_layers=DEF['num_layers'], dropout=DEF['dropout'])
            n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
            opt = optim.Adam(model.parameters(), lr=DEF['lr'])
            t0 = time.perf_counter()
            hist = train(model, tr, opt, nn.MSELoss(reduction='sum'), va, batch_size=DEF['batch_size'],
                         epochs=a.epocas, verbose=True, device=torch.device('cpu'))
            seg = time.perf_counter() - t0
            sc = scores(model, w_eval, S, C)
            pr = np.full(S, np.nan)
            for s in range(S):
                ys = y[:, s]
                v = ys >= 0
                if (ys[v] == 1).any() and (ys[v] == 0).any():
                    pr[s] = average_precision(ys[v], sc[v, s])
            ev = np.isfinite(pr)
            reg = dict(segundos=round(seg, 1), epocas=a.epocas, parametros=int(n_par), ventanas_train=len(tr),
                       pr_auc_media_bloques=float(np.nanmean(pr)) if ev.any() else None,
                       n_bloques_evaluables=int(ev.sum()), arq='si_dmseq', esquema=a.esquema,
                       canales=a.canales, pliegue=k, semilla='autores' if sem is None else sem,
                       perdida_train=[round(float(v), 2) for v in hist['train']],
                       perdida_val=[round(float(v), 2) for v in hist['val']])
            with open(os.path.join(SALIDA, 'resultados_g7.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(reg, ensure_ascii=False) + '\n')
            np.savez_compressed(os.path.join(SALIDA, etiqueta + '.npz'), score=sc.astype(np.float32),
                                y=y, pr_auc_bloque=pr, sensores=d['validos'], inicio=va_ini)
            torch.save(model.state_dict(), os.path.join(SALIDA, etiqueta + '.pth'))
            print(f'G7 {a.esquema} pliegue {k} semilla {reg["semilla"]}: {a.epocas} ep, {seg:.0f} s, '
                  f'PR-AUC medio por bloque {reg["pr_auc_media_bloques"]:.4f} ({reg["n_bloques_evaluables"]} bloques)',
                  flush=True)
        del tr, va, w_eval, Z, M

    print('\nListo. Resultados en resultados_g7\\resultados_g7.jsonl')


if __name__ == '__main__':
    main()
