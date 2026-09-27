"""Sonda: ¿la caída de EEGNet ante un hueco depende de DÓNDE cae el hueco?

Sobre la sesión 2 limpia (MOABB, mismos ensayos que el entrenamiento), borra en crudo
(0 V) un tramo de largo fijo al inicio, al medio o al final de la ventana [2, 6] s,
filtra como en inferencia y decodifica con cada modelo. CSP+LDA decodifica sobre las
muestras presentes (como en línea). Sin hueco = control.

Uso: python scripts/gap_position_probe.py --gap 1.52 --variants eegnet eegnet_gapaug ...
     → results/robustez/sonda_posicion.csv/.md
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from bcibench import data, robust as R  # noqa: E402
from bcibench.decoder import Decoder  # noqa: E402
from bcibench.eegnet import EEGNetDecoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gap", type=float, default=1.52)
ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
ap.add_argument("--variants", nargs="+", default=["eegnet", "eegnet_gapaug"])
ap.add_argument("--models", default="results/models")
ap.add_argument("--out", default="results/robustez")
a = ap.parse_args()
L = int(round(a.gap * R.SF))
POS = {"sin hueco": None, "inicio": 0, "medio": (R.N_WIN - L) // 2, "final": R.N_WIN - L}
rows = []
for s in a.subjects:
    X, y = R.raw_padded_epochs(data.load_subject(s), data.SESSION_TEST)
    csp = Decoder.load(os.path.join(a.models, f"decoder_s{s:02d}.pkl"))
    for pos, st in POS.items():
        n = len(X)
        start = np.full(n, 0 if st is None else st)
        Ls = np.full(n, 0 if st is None else L)
        Xg, mask = R.apply_gaps(X, start, Ls)
        W = R.window(R.bandpass(Xg)).astype(np.float32)
        present = mask[0] == 0
        rows.append(dict(subject=s, position=pos, decoder="csp",
                         bacc=balanced_accuracy_score(y, csp.model.predict(W[:, :, present].astype(np.float64)))))
        for v in a.variants:
            p = os.path.join(a.models, f"{v}_s{s:02d}.pkl")
            if not os.path.exists(p):
                continue
            net = EEGNetDecoder.load(p)
            Xin = np.concatenate([W, mask[:, None, :]], axis=1) if len(net.mu) == 23 else W
            rows.append(dict(subject=s, position=pos, decoder=v, bacc=balanced_accuracy_score(y, net.predict(Xin))))
    print("sujeto", s, flush=True)
D = pd.DataFrame(rows)
os.makedirs(a.out, exist_ok=True)
D.to_csv(os.path.join(a.out, "sonda_posicion.csv"), index=False)
T = D.pivot_table(index="decoder", columns="position", values="bacc", aggfunc="median")[list(POS)]
T["n"] = D.groupby("decoder").subject.nunique()
with open(os.path.join(a.out, "sonda_posicion.md"), "w", encoding="utf-8") as f:
    f.write(f"Sesión 2 limpia, hueco de {a.gap:g} s borrado en crudo; mediana entre sujetos de la balanced accuracy.\n\n")
    f.write(T.reset_index().to_markdown(index=False, floatfmt=".3f"))
print(T.to_string())
