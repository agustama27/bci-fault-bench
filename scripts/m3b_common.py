"""M3b · utilidades comunes (rutas, carga de ensayos emparejados con la referencia, estadística mínima).

No modifica nada del banco: solo lee campaign-2026-09-27/raw/ y analysis*/ y escribe en analysis-m3b/.
Análisis EXPLORATORIOS post hoc.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\agustin.tamagusuku\Desktop\bci-fault-bench\campaign-2026-09-27"
SRC = r"C:\Users\agustin.tamagusuku\Desktop\bci-fault-bench\src"
OUT = os.path.join(ROOT, "analysis-m3b")
os.makedirs(OUT, exist_ok=True)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

LABEL_ORDER = ["ref", "loss-random-0.01", "loss-random-0.05", "loss-random-0.1", "jitter-0.01", "jitter-0.05",
               "jitter-0.1", "delay-0.05", "delay-0.1", "delay-0.25", "disconnect-0.5", "disconnect-1",
               "disconnect-3", "burst_trial-0.1", "burst_trial-0.25", "burst_trial-0.4",
               "disconnect_trial-0.5", "disconnect_trial-1", "disconnect_trial-2"]
FAULT_LABELS = LABEL_ORDER[1:]
FAMILY = {"loss": "pérdida uniforme", "jitter": "jitter", "delay": "retraso", "disconnect": "desconexión uniforme",
          "burst_trial": "pérdida contigua en ensayo", "disconnect_trial": "desconexión en ensayo", "none": "referencia"}
DECODERS = {"CSP+LDA": "trials.csv", "EEGNet": "trials_eegnet.csv"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def wilcoxon_paired(a, b):
    """Wilcoxon de rangos con signo, apareado, bilateral. Devuelve (n_no_nulos, estadístico, p). NaN si no definido."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[m]
    nz = int((d != 0).sum())
    if nz == 0:
        return nz, np.nan, np.nan
    try:
        r = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
        return nz, float(r.statistic), float(r.pvalue)
    except ValueError:
        return nz, np.nan, np.nan


def load_paired(root: str = ROOT) -> pd.DataFrame:
    """Una fila por (ejecución con fallo o referencia, ensayo) emparejada con el MISMO ensayo (sujeto, corrida, índice)
    de la ejecución de referencia. Columnas por decodificador: pred_*, valid_*, ref_pred_*, ref_valid_*.

    n_samples (en línea, trials.csv) define el 'toque' para ambos decodificadores: es lo que recibió el pipeline.
    """
    execs = pd.read_csv(os.path.join(root, "analysis", "execs.csv"))
    rows = []
    ref_cache = {}
    for _, e in execs.iterrows():
        key = (int(e.subject), str(e.run))
        if key not in ref_cache:
            rd = os.path.join(root, "raw", f"s{key[0]:02d}-r{key[1]}-ref")
            ref_cache[key] = {dec: pd.read_csv(os.path.join(rd, f)) for dec, f in DECODERS.items()}
        d = os.path.join(root, "raw", e.exec_id)
        cur = {dec: pd.read_csv(os.path.join(d, f)) for dec, f in DECODERS.items()}
        t = cur["CSP+LDA"]
        r0 = ref_cache[key]["CSP+LDA"]
        assert len(t) == len(r0) == 24 and np.array_equal(t.label.to_numpy(), r0.label.to_numpy())
        assert np.abs(t.onset.to_numpy() - r0.onset.to_numpy()).max() < 1e-6, e.exec_id
        base = pd.DataFrame(dict(exec_id=e.exec_id, subject=int(e.subject), run=str(e.run), label_cond=e.label,
                                 kind=e.kind, severity=float(e.severity), i=np.arange(len(t)),
                                 onset=t.onset.to_numpy(), y=t.label.to_numpy(),
                                 n_samples=t.n_samples.to_numpy(), valid_online=t.valid.to_numpy()))
        for dec, tag in (("CSP+LDA", "csp"), ("EEGNet", "net")):
            f, r = cur[dec], ref_cache[key][dec]
            assert np.array_equal(f.label.to_numpy(), t.label.to_numpy())
            base[f"pred_{tag}"] = pd.to_numeric(f.pred, errors="coerce").to_numpy()
            base[f"valid_{tag}"] = f.valid.to_numpy()
            base[f"ref_pred_{tag}"] = pd.to_numeric(r.pred, errors="coerce").to_numpy()
            base[f"ref_valid_{tag}"] = r.valid.to_numpy()
        rows.append(base)
    D = pd.concat(rows, ignore_index=True)
    D["family"] = D.kind.map(FAMILY)
    D["touched"] = (D.n_samples < 1000) | (D.valid_online == 0)
    for tag in ("csp", "net"):
        v, rv = D[f"valid_{tag}"] == 1, D[f"ref_valid_{tag}"] == 1
        D[f"ok_{tag}"] = (v & (D[f"pred_{tag}"] == D.y)).astype(int)          # inválido = error
        D[f"ref_ok_{tag}"] = (rv & (D[f"ref_pred_{tag}"] == D.y)).astype(int)
        # cambio de decisión vs. la referencia: una decisión que desaparece (inválido) cuenta como cambio
        D[f"chg_{tag}"] = (~(v & rv & (D[f"pred_{tag}"] == D[f"ref_pred_{tag}"]))).astype(int)
        D[f"chg_valid_{tag}"] = np.where(v & rv, (D[f"pred_{tag}"] != D[f"ref_pred_{tag}"]).astype(float), np.nan)
    return D
