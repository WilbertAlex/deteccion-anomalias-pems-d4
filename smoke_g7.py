"""
Smoke-test de integracion de G7 (SI-DMSeq) - PPI Grupo 4, seccion 3.1 (gate semana 7).

Se ejecuta DESDE LA RAIZ del repositorio clonado de SI-DMSeq, con el entorno .venv-g7:
    python smoke_g7.py                 # prueba rapida (unos minutos)
    python smoke_g7.py --costo         # ademas mide el costo de una epoca a escala PeMS-D4 (291 sensores)

Que verifica (pasos 3 a 5 del smoke-test del PPI):
  A. Que el constructor vectorizado de ventanas produce EXACTAMENTE los mismos tensores que
     TrafficDataset de los autores (el original usa torch.cat en un bucle, O(n^2), inviable
     con ~13 000 ventanas de PeMS-D4). El modelo y el entrenamiento NO se modifican.
  B. Que el modelo de los autores (MultitaskSequenceModel) entrena con su funcion train()
     original y su configuracion por defecto (seq_len 24, embedding 64, 2 capas, dropout 0.2,
     Adam lr 0.01, MSE suma, lote 64, rho 0.25).
  C. Que se extrae el score puntual de la seccion 2.6.1: error de reconstruccion local del
     sensor s en el paso t (ultima posicion de la ventana), sin agregacion por red ni carta
     de control.
  D. Que ese score permite calcular PR-AUC por bloque sensor (average precision, misma
     definicion que sklearn.average_precision_score).

Los datos de esta prueba son SINTETICOS con forma de PeMS-D4 (pasos x sensores, flujo
normalizado con patron diario). No es un resultado del estudio: solo verifica la tuberia.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
try:
    from src.data_utils import TrafficDataset
    from src.model import MultitaskSequenceModel
    from src.train import train
except ImportError as e:  # pragma: no cover
    sys.exit(f"No se encuentra el codigo de los autores ({e}). Ejecuta el script desde D:\\ppi\\SI-DMSeq")

# Configuracion por defecto publicada (src/train.py, bloque __main__)
DEF = dict(seq_len=24, y_len=1, embedding_size=64, num_layers=2, dropout=0.2,
           lr=0.01, batch_size=64, epochs=20, rho=0.25, train_size=0.8)


class FastTrafficDataset(TrafficDataset):
    """Mismo resultado que TrafficDataset, construido con unfold (sin bucle de torch.cat).

    Recibe una matriz (pasos, sensores) en lugar de CSV. Reproduce paso a paso el original:
    semilla 0, dropout de entrada con rho (solo en train), ventanas de seq_len y objetivo de
    y_len, mascara de ceros, relleno con la media por instante y corte train/val por train_size.
    """

    def __init__(self, matrices, seq_len, y_len, feature_size, mode='train', rho=0.25, train_size=0.8):
        self.rho = rho if mode == 'train' else .0
        torch.manual_seed(0)
        seqs, targets = [], []
        for m in matrices:
            comp = torch.as_tensor(np.asarray(m, dtype=np.float32))
            incomp = F.dropout(comp, p=self.rho) * (1 - self.rho)
            n = comp.shape[0] - seq_len - y_len + 1
            # unfold(0, size, 1) -> (n_ventanas, sensores, size); se reordena a (n, size, sensores)
            seqs.append(incomp[:n + seq_len - 1].unfold(0, seq_len, 1).permute(0, 2, 1)[:n].contiguous())
            targets.append(incomp[seq_len:].unfold(0, y_len, 1).permute(0, 2, 1)[:n].contiguous())
        self.seqs = torch.cat(seqs, dim=0)
        self.targets = torch.cat(targets, dim=0)
        self.seq_masks = (self.seqs == 0)
        mean = (self.seqs.sum(dim=2) / (~self.seq_masks).sum(dim=2)).unsqueeze(2).repeat(1, 1, feature_size)
        self.seqs += self.seq_masks * mean
        self.seq_len = seq_len
        N = int(train_size * len(self.seqs))
        sl = slice(None, N) if mode == 'train' else slice(N, None)
        self.seqs, self.seq_masks, self.targets = self.seqs[sl], self.seq_masks[sl], self.targets[sl]
        self.target_masks = (self.targets == 0)


def average_precision(y, s):
    """PR-AUC como sklearn.metrics.average_precision_score (umbrales distintos, sin interpolar)."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    orden = np.argsort(-s, kind='mergesort')
    y, s = y[orden], s[orden]
    corte = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[corte]
    fp = (corte + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def serie_sintetica(pasos, sensores, semilla=7):
    """Flujo normalizado con patron diario de 288 pasos (5 min), ruido y algun cero real."""
    rng = np.random.default_rng(semilla)
    t = np.arange(pasos)[:, None]
    fase = rng.uniform(0, 2 * np.pi, size=(1, sensores))
    x = np.sin(2 * np.pi * t / 288 + fase) + 0.3 * np.sin(4 * np.pi * t / 288 + fase) \
        + 0.15 * rng.standard_normal((pasos, sensores))
    x = (x - x.mean(0)) / x.std(0)
    x[rng.random(x.shape) < 0.001] = 0.0
    return x.astype(np.float32)


def inyectar(x, desde, prevalencia=0.03, semilla=42):
    """Picos puntuales (+/- 4 desviaciones) solo desde 'desde'. Devuelve serie y etiquetas."""
    rng = np.random.default_rng(semilla)
    y = np.zeros_like(x, dtype=np.int8)
    x = x.copy()
    for s in range(x.shape[1]):
        k = max(1, int(prevalencia * (x.shape[0] - desde)))
        pos = rng.choice(np.arange(desde, x.shape[0]), size=k, replace=False)
        x[pos, s] += rng.choice([-4.0, 4.0], size=k)
        y[pos, s] = 1
    return x, y


@torch.no_grad()
def score_local(model, seqs, lote=256):
    """Seccion 2.6.1: error de reconstruccion local por sensor en la ultima posicion de la ventana."""
    model.eval()
    out = []
    for i in range(0, len(seqs), lote):
        x = seqs[i:i + lote]
        recon, _ = model(x)
        out.append(((recon[:, -1, :] - x[:, -1, :]) ** 2).numpy())
    return np.concatenate(out, axis=0)  # (ventanas, sensores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensores', type=int, default=20)
    ap.add_argument('--pasos', type=int, default=3000)
    ap.add_argument('--epocas', type=int, default=3, help='la configuracion publicada usa 20')
    ap.add_argument('--costo', action='store_true', help='mide una epoca a escala PeMS-D4')
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    informe = {'torch': torch.__version__, 'numpy': np.__version__, 'config_publicada': DEF}

    # ---------- A. Equivalencia con TrafficDataset de los autores ----------
    print('[A] Equivalencia del constructor de ventanas con TrafficDataset original...')
    import tempfile
    import pandas as pd
    xa = serie_sintetica(220, 6)
    with tempfile.TemporaryDirectory() as d:
        pd.DataFrame(xa).to_csv(os.path.join(d, 'p.csv'), index=False)
        xa = pd.read_csv(os.path.join(d, 'p.csv'), dtype='float32').values   # misma lectura que el original
        for modo in ('train', 'val'):
            orig = TrafficDataset(d, ['p.csv'], DEF['seq_len'], DEF['y_len'], 6, mode=modo, rho=DEF['rho'])
            rap = FastTrafficDataset([xa], DEF['seq_len'], DEF['y_len'], 6, mode=modo, rho=DEF['rho'])
            for att in ('seqs', 'seq_masks', 'targets', 'target_masks'):
                a_, b_ = getattr(orig, att), getattr(rap, att)
                assert a_.shape == b_.shape, f'{modo}.{att}: forma {a_.shape} vs {b_.shape}'
                assert torch.equal(a_, b_), f'{modo}.{att}: valores distintos'
    print('    OK: seqs, mascaras y objetivos identicos bit a bit (train y val).')
    informe['A_equivalencia'] = 'ok'

    # ---------- B. Entrenamiento con train() y configuracion de los autores ----------
    S, T = a.sensores, a.pasos
    print(f'[B] Entrenamiento con la configuracion publicada: {S} sensores, {T} pasos, {a.epocas} epocas...')
    x = serie_sintetica(T, S)
    n_win = T - DEF['seq_len'] - DEF['y_len'] + 1
    corte = int(DEF['train_size'] * n_win)
    inicio_val = corte + DEF['seq_len'] - 1          # primer instante final de ventana en val
    x_iny, y_iny = inyectar(x, desde=inicio_val + 1)
    tr = FastTrafficDataset([x_iny], DEF['seq_len'], DEF['y_len'], S, 'train', DEF['rho'])
    va = FastTrafficDataset([x_iny], DEF['seq_len'], DEF['y_len'], S, 'val', DEF['rho'])
    torch.manual_seed(1)
    model = MultitaskSequenceModel(feature_size=S, embedding_size=DEF['embedding_size'],
                                   num_layers=DEF['num_layers'], dropout=DEF['dropout'])
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = optim.Adam(model.parameters(), lr=DEF['lr'])
    t0 = time.perf_counter()
    hist = train(model, tr, opt, nn.MSELoss(reduction='sum'), va,
                 batch_size=DEF['batch_size'], epochs=a.epocas, verbose=True, device=torch.device('cpu'))
    seg = time.perf_counter() - t0
    assert all(math.isfinite(v) for v in hist['train'] + hist['val']), 'perdida no finita'
    print(f'    OK: {seg:.1f} s, {n_par:,} parametros, perdida train {hist["train"][0]:.1f} -> {hist["train"][-1]:.1f}')
    informe['B_entrenamiento'] = dict(segundos=round(seg, 1), parametros=n_par, perdida_train=hist['train'],
                                      perdida_val=hist['val'])

    # ---------- C. Score local por sensor e instante ----------
    print('[C] Extraccion del score local (seccion 2.6.1)...')
    sc = score_local(model, va.seqs)
    t_fin = np.arange(len(va.seqs)) + corte + DEF['seq_len'] - 1   # instante final de cada ventana
    assert sc.shape == (len(va.seqs), S) and np.isfinite(sc).all()
    etiquetas = y_iny[t_fin, :]
    print(f'    OK: score de forma {sc.shape} (ventanas x sensores), sin NaN.')

    # ---------- D. PR-AUC por bloque sensor ----------
    print('[D] PR-AUC por bloque (sensor)...')
    pr = np.array([average_precision(etiquetas[:, s], sc[:, s]) for s in range(S)])
    azar = etiquetas.mean()
    print(f'    OK: PR-AUC medio {np.nanmean(pr):.3f} (azar {azar:.3f}), bloques evaluables {np.isfinite(pr).sum()}/{S}')
    informe['D_pr_auc'] = dict(media=float(np.nanmean(pr)), azar=float(azar), evaluables=int(np.isfinite(pr).sum()))

    # ---------- Costo a escala PeMS-D4 ----------
    if a.costo:
        S2, T2 = 291, 13593      # 291 sensores; 80 % inicial = 13 593 pasos
        print(f'[E] Costo de UNA epoca a escala PeMS-D4 ({S2} sensores, {T2} pasos, bloque 80 %)...')
        x2 = serie_sintetica(T2, S2)
        tr2 = FastTrafficDataset([x2], DEF['seq_len'], DEF['y_len'], S2, 'train', DEF['rho'], train_size=1.0)
        va2 = FastTrafficDataset([x2[-500:]], DEF['seq_len'], DEF['y_len'], S2, 'val', DEF['rho'], train_size=0.0)
        torch.manual_seed(1)
        m2 = MultitaskSequenceModel(S2, DEF['embedding_size'], DEF['num_layers'], DEF['dropout'])
        t0 = time.perf_counter()
        train(m2, tr2, optim.Adam(m2.parameters(), lr=DEF['lr']), nn.MSELoss(reduction='sum'), va2,
              batch_size=DEF['batch_size'], epochs=1, verbose=True, device=torch.device('cpu'))
        ep = time.perf_counter() - t0
        tot = ep * DEF['epochs']
        print(f'    Una epoca: {ep:.0f} s -> 20 epocas: {tot/60:.0f} min por corrida (cota alta: pliegue mas largo)')
        informe['E_costo'] = dict(segundos_epoca=round(ep, 1), minutos_20_epocas=round(tot / 60, 1))

    with open('smoke_g7_resultado.json', 'w', encoding='utf-8') as f:
        json.dump(informe, f, indent=1, ensure_ascii=False)
    print('\nSMOKE-TEST G7: TUBERIA VERIFICADA. Resultado en smoke_g7_resultado.json')


if __name__ == '__main__':
    main()
