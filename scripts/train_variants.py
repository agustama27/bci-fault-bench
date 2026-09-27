"""Entrena variantes endurecidas de EEGNet por sujeto (línea futura 4).

Mismo protocolo que `train_eegnet.py`: sesión "0train", 300 épocas, semilla 0,
Adam 1e-3, lote 32, misma arquitectura. Diferencias:
  - la entrada se arma como en inferencia: época cruda [1, 7) s → (hueco en 0 V)
    → filtro 8-30 Hz → recorte [2, 6) s (ver bcibench.robust);
  - en cada época, cada ensayo recibe con probabilidad p un tramo contiguo borrado
    de largo U(lo, hi) s en posición uniforme dentro de la ventana;
  - --mask agrega el canal 23 "muestra faltante".

Uso: python scripts/train_variants.py --variant gapaug --subjects 1 2 3 [--p 0.5]
     → results/models/eegnet_<variant>_sXX.pkl, results/train_<variant>.csv
"""
import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from bcibench import data, robust as R  # noqa: E402
from bcibench.eegnet import EEGNetDecoder  # noqa: E402

CONF = {  # variante → (lo_s, hi_s, máscara)
    "gapaug": (0.4, 1.6, False),
    "gapaug_wide": (0.4, 2.6, False),
    "mask": (0.4, 1.6, True),
}
ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=list(CONF))
ap.add_argument("--subjects", nargs="+", type=int, default=[1])
ap.add_argument("--p", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--models", default="results/models")
a = ap.parse_args()
lo, hi, use_mask = CONF[a.variant]
os.makedirs(a.models, exist_ok=True)
out_csv = f"results/train_{a.variant}.csv"
for s in a.subjects:
    t0 = time.time()
    sd = data.load_subject(s)
    Xtr, ytr = R.raw_padded_epochs(sd, data.SESSION_TRAIN)
    Xte, yte = R.raw_padded_epochs(sd, data.SESSION_TEST)
    t_load = time.time() - t0
    dec = R.train_gapaug(s, Xtr, ytr, p=a.p, lo_s=lo, hi_s=hi, use_mask=use_mask, epochs=a.epochs,
                         log=lambda m: print(m, flush=True))
    t_train = time.time() - t0 - t_load
    Wte = R.window(R.bandpass(Xte)).astype(np.float32)
    if use_mask:
        Wte = np.concatenate([Wte, np.zeros((len(Wte), 1, R.N_WIN), np.float32)], axis=1)
    bacc = balanced_accuracy_score(yte, dec.predict(Wte))
    orig = EEGNetDecoder.load(os.path.join(a.models, f"eegnet_s{s:02d}.pkl"))
    bacc_orig = balanced_accuracy_score(yte, orig.predict(Wte[:, :22]))   # mismo insumo, modelo original
    dec.save(os.path.join(a.models, f"eegnet_{a.variant}_s{s:02d}.pkl"))
    row = dict(variant=a.variant, subject=s, p=a.p, gap_lo_s=lo, gap_hi_s=hi, mask=int(use_mask),
               epochs=a.epochs, n_train=len(ytr), bacc_clean_session2=round(bacc, 4), bacc_clean_session2_orig=round(bacc_orig, 4),
               seconds_load=round(t_load), seconds_train=round(t_train))
    print(row, flush=True)
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
