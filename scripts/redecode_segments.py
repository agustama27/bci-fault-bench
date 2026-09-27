"""Re-decodifica fuera de línea los segmentos guardados por el consumidor con otro
decodificador (EEGNet) y, como control, con el mismo CSP+LDA en línea.

Equivalencia: el decodificador es determinista y el segmento es exactamente lo que el
pipeline recibió (con sus huecos y su temporalidad), de modo que la predicción fuera de
línea es la que habría dado ese decodificador enchufado en vivo.

Grilla nominal: la ventana [2, 6] s se reconstruye a 250 Hz (1000 muestras) ubicando cada
muestra recibida por su marca de tiempo; las posiciones sin muestra quedan en cero
(una BCI real debe entregar un tensor de tamaño fijo). El filtro 8-30 Hz se aplica sobre
la ventana con relleno de 1 s y luego se recorta, como en línea. Validez: misma regla que
en línea (al menos medio segundo de señal, 125 muestras, dentro de la ventana).

Uso: python scripts/redecode_segments.py --root results/vm/campana2 --models results/models
     → <exec>/trials_eegnet.csv y <exec>/trials_csp_offline.csv
"""
import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mne.filter import filter_data  # noqa: E402

from bcibench import data  # noqa: E402
from bcibench.decoder import Decoder  # noqa: E402
from bcibench.eegnet import EEGNetDecoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

SF = data.SFREQ
TMIN, TMAX, PAD = data.TMIN, data.TMAX, 1.0
N_WIN = int(round((TMAX - TMIN) * SF))          # 1000
N_PAD = int(round((TMAX - TMIN + 2 * PAD) * SF))  # 1500


def grid(ts_rel: np.ndarray, x: np.ndarray, n_ch: int):
    """Coloca las muestras recibidas en la grilla nominal [TMIN-PAD, TMAX+PAD)."""
    g = np.zeros((n_ch, N_PAD), dtype=np.float64)
    idx = np.round((ts_rel - (TMIN - PAD)) * SF).astype(int)
    ok = (idx >= 0) & (idx < N_PAD)
    g[:, idx[ok]] = x[ok].T
    present = np.zeros(N_PAD, dtype=bool)
    present[idx[ok]] = True
    return g, present


dec_cache = {}


def decoders(subject: int):
    if subject not in dec_cache:
        csp = Decoder.load(os.path.join(a.models, f"decoder_s{subject:02d}.pkl"))
        net_p = os.path.join(a.models, f"eegnet_s{subject:02d}.pkl")
        net = EEGNetDecoder.load(net_p) if os.path.exists(net_p) else None
        dec_cache[subject] = (csp, net)
    return dec_cache[subject]


n_done = 0
for d in sorted(glob.glob(os.path.join(a.root, "s*/"))):
    seg_p = os.path.join(d, "segments.npz")
    if not os.path.exists(seg_p):
        continue
    out_net, out_csp = os.path.join(d, "trials_eegnet.csv"), os.path.join(d, "trials_csp_offline.csv")
    if os.path.exists(out_net) and not a.force:
        continue
    subject = int(os.path.basename(d.rstrip("/\\")).split("-")[0][1:])
    csp, net = decoders(subject)
    z = np.load(seg_p)
    meta = z["meta"]
    n_ch = len(csp.model.named_steps["csp"].patterns_) if hasattr(csp.model.named_steps["csp"], "patterns_") else 22
    rows_net, rows_csp = [], []
    for k in range(len(meta)):
        onset, label, valid_online = meta[k]
        ts, x = z[f"t{k}_ts"], z[f"t{k}_x"]
        rec = dict(onset=round(float(onset), 4), label=int(label))
        if len(ts) == 0:
            for rows in (rows_net, rows_csp):
                rows.append(dict(rec, pred="", proba="", n_samples=0, valid=0))
            continue
        g, present = grid(ts.astype(float), x.astype(np.float64) * 1e-6, x.shape[1])
        n_in = int(present[int(PAD * SF):int(PAD * SF) + N_WIN].sum())
        valid = int(n_in >= int(0.5 * SF))   # misma regla que en línea: al menos medio segundo (125 muestras) en la ventana
        gf = filter_data(g, SF, data.FMIN, data.FMAX, method="iir", verbose=False)
        win = gf[:, int(PAD * SF):int(PAD * SF) + N_WIN]
        if not valid:
            rows_net.append(dict(rec, pred="", proba="", n_samples=n_in, valid=0))
            rows_csp.append(dict(rec, pred="", proba="", n_samples=n_in, valid=0))
            continue
        # CSP fuera de línea sobre las muestras presentes (como en línea: sin ceros)
        m = present[int(PAD * SF):int(PAD * SF) + N_WIN]
        pc = csp.predict_proba(win[:, m][None])[0]
        rows_csp.append(dict(rec, pred=int(csp.model.classes_[int(pc.argmax())]), proba=round(float(pc.max()), 4), n_samples=n_in, valid=1))
        if net is not None:
            pn = net.predict_proba(win[None].astype(np.float32))[0]
            rows_net.append(dict(rec, pred=int(net.classes[int(pn.argmax())]), proba=round(float(pn.max()), 4), n_samples=n_in, valid=1))
    for path, rows in ((out_net, rows_net), (out_csp, rows_csp)):
        if rows:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["onset", "label", "pred", "proba", "n_samples", "valid"])
                w.writeheader(); w.writerows(rows)
    n_done += 1
    if n_done % 50 == 0:
        print(n_done, flush=True)
print("re-decodificadas:", n_done)
