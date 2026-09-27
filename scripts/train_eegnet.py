"""Entrena EEGNet por sujeto (sesión 1) y lo evalúa fuera de línea (sesión 2), como PoC 1.

Uso: python scripts/train_eegnet.py [sujetos...]   → results/models/eegnet_sXX.pkl, results/eegnet_offline.csv
"""
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from bcibench import data, eegnet  # noqa: E402

subjects = [int(s) for s in sys.argv[1:]] or [1]
os.makedirs("results/models", exist_ok=True)
rows = []
for s in subjects:
    t0 = time.time()
    sd = data.load_subject(s)
    tr = data.session_xy(sd, data.SESSION_TRAIN)
    te = data.session_xy(sd, data.SESSION_TEST)
    n = min(tr.X.shape[2], te.X.shape[2], 1000)
    dec = eegnet.train(s, tr.X[:, :, :n], tr.y, epochs=300, X_val=te.X[:, :, :n], y_val=te.y, verbose=True)
    pred = dec.predict(te.X[:, :, :n])
    bacc = balanced_accuracy_score(te.y, pred)
    r01 = data.session_xy(sd, data.SESSION_TEST, ["0", "1"])
    b01 = balanced_accuracy_score(r01.y, dec.predict(r01.X[:, :, :n]))
    dec.save(f"results/models/eegnet_s{s:02d}.pkl")
    rows.append(dict(subject=s, eegnet_all=round(bacc, 4), eegnet_runs01=round(b01, 4), seconds=round(time.time() - t0)))
    print(rows[-1], flush=True)
with open("results/eegnet_offline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subject", "eegnet_all", "eegnet_runs01", "seconds"]); w.writeheader(); w.writerows(rows)
