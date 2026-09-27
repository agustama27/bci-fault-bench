"""Re-decodifica los segmentos de la campaña con variantes de EEGNet (líneas futuras 4 y 5).

Igual que `redecode_segments.py` (grilla nominal, filtro 8-30 Hz con 1 s de relleno,
recorte [2, 6] s, regla de validez de 125 muestras en la ventana), pero
parametrizado por variante = (patrón de modelo, relleno de huecos, canal máscara).

NO escribe en la carpeta de la campaña (solo lectura). Crea un espejo
`<out>/<exec>/` con COPIAS de producer.json, consumer.json, telemetry.csv,
trials.csv y trials_csp_offline.csv (y el trials_eegnet.csv original como
trials_eegnet_orig.csv, para el control), más:
  trials_<variante>.csv   mismas columnas que trials_eegnet.csv
  trials_sentinel.csv     señales del vigía por ensayo: n_samples, max_gap_ms

Las ventanas filtradas se calculan una vez por ensayo y por tipo de relleno y se
reusan entre variantes. Si falta el modelo de un sujeto para una variante, esa
ejecución queda sin archivo de esa variante (analyze.py la omite).

Uso: python scripts/redecode_variants.py --root <campana2> --models results/models \
        --out results/campana2-eval --variants eegnet eegnet_fill_interp ...
"""
import argparse
import csv
import glob
import os
import shutil
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bcibench import robust as R  # noqa: E402
from bcibench.eegnet import EEGNetDecoder  # noqa: E402

# nombre → (prefijo del archivo de modelo, relleno, máscara)
VARIANTS = {
    "eegnet": ("eegnet", "zero", False),
    "eegnet_fill_interp": ("eegnet", "interp", False),
    "eegnet_fill_hold": ("eegnet", "hold", False),
    "eegnet_gapaug": ("eegnet_gapaug", "zero", False),
    "eegnet_gapaug_fill_interp": ("eegnet_gapaug", "interp", False),
    "eegnet_gapaug_wide": ("eegnet_gapaug_wide", "zero", False),
    "eegnet_mask": ("eegnet_mask", "zero", True),
}
COPY = ["producer.json", "consumer.json", "telemetry.csv", "trials.csv", "trials_csp_offline.csv"]
FIELDS = ["onset", "label", "pred", "proba", "n_samples", "valid"]

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variants", nargs="+", default=["eegnet"], choices=list(VARIANTS))
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

MIN_VALID = int(0.5 * R.SF)   # 125 muestras en la ventana (misma regla que en línea)
models = {}


def model(prefix: str, subject: int):
    key = (prefix, subject)
    if key not in models:
        p = os.path.join(a.models, f"{prefix}_s{subject:02d}.pkl")
        models[key] = EEGNetDecoder.load(p) if os.path.exists(p) else None
    return models[key]


def max_gap_ms(present_win: np.ndarray) -> float:
    """Mayor intervalo entre muestras presentes en la ventana, contando los bordes."""
    pos = np.flatnonzero(present_win)
    if len(pos) == 0:
        return float(len(present_win) + 1) * 1e3 / R.SF
    ext = np.concatenate([[-1], pos, [len(present_win)]])
    return float(np.diff(ext).max()) * 1e3 / R.SF


def write(path, rows, fields=FIELDS):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


t_start = time.time()
n_done = 0
for d in sorted(glob.glob(os.path.join(a.root, "s*/"))):
    seg_p = os.path.join(d, "segments.npz")
    if not os.path.exists(seg_p):
        continue
    eid = os.path.basename(d.rstrip("/\\"))
    od = os.path.join(a.out, eid)
    os.makedirs(od, exist_ok=True)
    for f in COPY:
        if not os.path.exists(os.path.join(od, f)) and os.path.exists(os.path.join(d, f)):
            shutil.copy2(os.path.join(d, f), os.path.join(od, f))
    if not os.path.exists(os.path.join(od, "trials_eegnet_orig.csv")) and os.path.exists(os.path.join(d, "trials_eegnet.csv")):
        shutil.copy2(os.path.join(d, "trials_eegnet.csv"), os.path.join(od, "trials_eegnet_orig.csv"))
    subject = int(eid.split("-")[0][1:])
    todo = [v for v in a.variants
            if (a.force or not os.path.exists(os.path.join(od, f"trials_{v}.csv"))) and model(VARIANTS[v][0], subject) is not None]
    need_sentinel = a.force or not os.path.exists(os.path.join(od, "trials_sentinel.csv"))
    if not todo and not need_sentinel:
        continue
    z = np.load(seg_p)
    meta = z["meta"]
    recs, grids, pres, nin, valid = [], [], [], [], []
    for k in range(len(meta)):
        onset, label, _ = meta[k]
        ts, x = z[f"t{k}_ts"], z[f"t{k}_x"]
        recs.append(dict(onset=round(float(onset), 4), label=int(label)))
        if len(ts) == 0:
            g, p = np.zeros((22, R.N_PAD)), np.zeros(R.N_PAD, bool)
        else:
            g, p = R.grid(ts.astype(float), x.astype(np.float64) * 1e-6, x.shape[1])
        grids.append(g); pres.append(p)
        n_in = int(p[R.W0:R.W0 + R.N_WIN].sum())
        nin.append(n_in)
        valid.append(len(ts) > 0 and n_in >= MIN_VALID)
    valid = np.array(valid)
    vidx = np.flatnonzero(valid)
    if need_sentinel:
        write(os.path.join(od, "trials_sentinel.csv"),
              [dict(r, n_samples=nin[i], max_gap_ms=round(max_gap_ms(pres[i][R.W0:R.W0 + R.N_WIN]), 1), valid=int(valid[i]))
               for i, r in enumerate(recs)], ["onset", "label", "n_samples", "max_gap_ms", "valid"])
    wins = {}   # relleno → (n_valid, 22, 1000): se filtra una vez por ensayo y por relleno
    for v in todo:
        prefix, how, use_mask = VARIANTS[v]
        net = model(prefix, subject)
        rows = [dict(r, pred="", proba="", n_samples=nin[i], valid=0) for i, r in enumerate(recs)]
        if len(vidx):
            if how not in wins:
                G = np.stack([R.fill(grids[i], pres[i], how) for i in vidx])
                wins[how] = R.window(R.bandpass(G)).astype(np.float32)
            X = wins[how]
            if use_mask:
                M = np.stack([(~pres[i][R.W0:R.W0 + R.N_WIN]).astype(np.float32) for i in vidx])[:, None, :]
                X = np.concatenate([X, M], axis=1)
            P = net.predict_proba(X)
            for j, i in enumerate(vidx):
                rows[i].update(pred=int(net.classes[int(P[j].argmax())]), proba=round(float(P[j].max()), 4), valid=1)
        write(os.path.join(od, f"trials_{v}.csv"), rows)
    n_done += 1
    if n_done % 100 == 0:
        print(n_done, f"{time.time() - t_start:.0f}s", flush=True)
print("re-decodificadas:", n_done, f"en {time.time() - t_start:.0f}s")
