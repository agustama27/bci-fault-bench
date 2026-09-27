"""PoC 1 — clasificación offline (sanity check del loader y el decodificador).

Entrena CSP+LDA en la sesión 0train y evalúa en 1test, por sujeto.
Uso: python scripts/poc1_offline.py [sujetos...]   (por defecto: 1)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bcibench import data, decoder  # noqa: E402

subjects = [int(s) for s in sys.argv[1:]] or [1]
os.makedirs("results/poc1", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

for s in subjects:
    t0 = time.time()
    sd = data.load_subject(s)
    tr = data.session_xy(sd, data.SESSION_TRAIN)
    te = data.session_xy(sd, data.SESSION_TEST)
    dec = decoder.train(s, tr.X, tr.y)
    res = decoder.evaluate(dec, te.X, te.y)
    res.update(subject=s, n_train=len(tr), seconds=round(time.time() - t0, 1))
    dec.save(f"results/models/decoder_s{s:02d}.pkl")
    with open(f"results/poc1/s{s:02d}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))
