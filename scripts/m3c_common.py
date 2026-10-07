"""M3c · utilidades comunes de la CAMPAÑA EXPLORATORIA POST HOC (campaign-m3/campana3, 387 ejecuciones).

No modifica el banco ni las campañas: solo lee campaign-m3/campana3/ y campaign-2026-09-27/{raw,analysis}/ y escribe
en campaign-m3/analysis/. Reutiliza bcibench.metrics / bcibench.stats (mismas definiciones del estudio).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

BENCH = r"C:\Users\agustin.tamagusuku\Desktop\bci-fault-bench"
SRC = os.path.join(BENCH, "src")
NEW = os.path.join(BENCH, "campaign-m3", "campana3")
OLD_ROOT = os.path.join(BENCH, "campaign-2026-09-27")
OLD = os.path.join(OLD_ROOT, "raw")
OUT = os.path.join(BENCH, "campaign-m3", "analysis")
os.makedirs(OUT, exist_ok=True)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bcibench import metrics as M  # noqa: E402
from bcibench import stats as S  # noqa: E402

SFREQ = 250.0
NET_NEW = "trials_eegnet_ceros.csv"   # EEGNet estándar (ceros) sobre segments.npz de la campaña nueva (m3_eegnet_relleno.py)
NET_OLD = "trials_eegnet.csv"         # EEGNet del Entregable 2 sobre la campaña anterior

SWEEP = ["ref", "disconnect-n2-1", "disconnect-n5-1", "disconnect-n10-1", "disconnect-n20-1"]
SWEEP_K = {"ref": 0, "disconnect-n2-1": 2, "disconnect-n5-1": 5, "disconnect-n10-1": 10, "disconnect-n20-1": 20}
HOLD = ["ref", "hold_trial-0.1", "hold_trial-0.25", "hold_trial-0.4"]
BURST = ["burst_trial-0.1", "burst_trial-0.25", "burst_trial-0.4"]
FLOOR = ["disconnect-n5-1.25", "disconnect-n5-1.75", "disconnect-n5-2.25"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


# ---------------------------------------------------------------- carga de ejecuciones
def load(root: str, trials_file: str = "trials.csv", keep=None):
    """(execs, trials, tele): igual que bcibench.metrics.load_campaign pero solo para las ejecuciones cuyo
    directorio cumple `keep(nombre_de_carpeta) -> bool` (misma lógica y mismas funciones de métricas)."""
    import glob
    execs, trials, tele = [], {}, {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d) or (keep is not None and not keep(os.path.basename(d))):
            continue
        pj, cj = os.path.join(d, "producer.json"), os.path.join(d, "consumer.json")
        tp = os.path.join(d, trials_file)
        if not (os.path.exists(pj) and os.path.exists(cj) and os.path.exists(tp)):
            continue
        p, c = json.load(open(pj)), json.load(open(cj))
        eid = p["exec_id"]
        tr = pd.read_csv(tp)
        if "decision_delay" not in tr.columns:
            tr["decision_delay"] = np.nan
        te = pd.read_csv(os.path.join(d, "telemetry.csv"))
        te = te[te["sec"] < int(np.ceil(p["duration_s"]))].reset_index(drop=True)
        trials[eid], tele[eid] = tr, te
        f = p["fault"]
        execs.append(dict(exec_id=eid, subject=p["subject"], run=str(p["run"]), kind=f["kind"], severity=f["severity"],
                          mode=f["mode"], label=p["label"], n_pushed=p["n_pushed"], n_dropped=p["n_dropped"],
                          outages=len(p["outages"]), timing_p99_ms=p["timing_err_ms"]["p99"], n_trials=c["n_trials"],
                          n_valid=c["n_valid"], exceptions=c["exceptions"], **M.integrity_metrics(te, p),
                          **M.functional_metrics(tr)))
    return pd.DataFrame(execs), trials, tele


def lab_filter(labels):
    """keep() por etiqueta de condición: la carpeta es sXX-rY-<label>."""
    labels = set(labels)
    return lambda name: name.split("-", 2)[2] in labels if name.count("-") >= 2 else False


def load_events(path):
    ev = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return ev


# ---------------------------------------------------------------- pruebas por grupo (mismo código que el estudio)
def tests_group(med: pd.DataFrame, metric: str, kind: str, label_map: dict | None = None) -> pd.DataFrame:
    """Friedman + Wilcoxon apareado (Holm DENTRO del tipo) + r, usando bcibench.stats.by_kind_tests.

    `med`: mediana por sujeto con columnas subject, kind, severity, label. Se pasa `kind` como único tipo de fallo
    (bcibench.stats itera sobre su lista global KINDS; se sustituye temporalmente por [kind]).
    """
    old = S.KINDS
    S.KINDS = [kind]
    try:
        t = S.by_kind_tests(med, metric)
    finally:
        S.KINDS = old
    return t


def subject_med(df: pd.DataFrame, cols) -> pd.DataFrame:
    return M.subject_median(df, cols)


def wilcoxon_diff(a: pd.Series, b: pd.Series) -> dict:
    """Wilcoxon apareado a-b (alineado por índice = sujeto)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    t = S.wilcoxon_paired(j.a.to_numpy(), j.b.to_numpy())
    t["n"] = int(len(j))
    return t


# ---------------------------------------------------------------- ensayos emparejados con la referencia
def interval_hits(exec_dir: str, onset: np.ndarray, w_start=2.0, w_end=6.0):
    """Solape (en muestras) entre cada ventana de decodificación y los interval_planned de fault_log (si existen)."""
    p = os.path.join(exec_dir, "fault_log.jsonl")
    iv = [(e["start_sample"], e["end_sample"]) for e in load_events(p) if e.get("event") == "interval_planned"]
    ov = np.zeros(len(onset), dtype=int)
    inside = np.zeros(len(onset), dtype=int)
    for k, o in enumerate(onset):
        a, b = (o + w_start) * SFREQ, (o + w_end) * SFREQ
        for s, e in iv:
            x = max(0.0, min(b, e) - max(a, s))
            ov[k] += int(round(x))
            if s >= a - 1e-6 and e <= b + 1e-6:
                inside[k] += 1
    return ov, inside, len(iv)


def paired_trials(execs: pd.DataFrame, raw: str, net_file: str, ref_raw: str, ref_net_file: str,
                  with_intervals: bool = False) -> pd.DataFrame:
    """Una fila por (ejecución de `execs`, ensayo), emparejada con el MISMO ensayo (sujeto, corrida, índice, onset) de la
    ejecución de referencia en `ref_raw`. Columnas: pred_/valid_/ok_/chg_ para CSP+LDA ('csp') y EEGNet ('net').

    ok = decisión correcta (inválido cuenta como error); chg = cambio de decisión vs la referencia (una decisión que
    desaparece, es decir un inválido que antes era válido, cuenta como cambio).
    """
    rows, cache = [], {}
    for _, e in execs.iterrows():
        key = (int(e.subject), str(e.run))
        if key not in cache:
            rd = os.path.join(ref_raw, f"s{key[0]:02d}-r{key[1]}-ref")
            cache[key] = {"csp": pd.read_csv(os.path.join(rd, "trials.csv")),
                          "net": pd.read_csv(os.path.join(rd, ref_net_file))}
        d = os.path.join(raw, e.exec_id)
        cur = {"csp": pd.read_csv(os.path.join(d, "trials.csv")), "net": pd.read_csv(os.path.join(d, net_file))}
        t, r0 = cur["csp"], cache[key]["csp"]
        assert len(t) == len(r0) == 24 and np.array_equal(t.label.to_numpy(), r0.label.to_numpy()), e.exec_id
        assert np.abs(t.onset.to_numpy() - r0.onset.to_numpy()).max() < 1e-6, e.exec_id
        base = pd.DataFrame(dict(exec_id=e.exec_id, subject=int(e.subject), run=str(e.run), label_cond=e.label,
                                 kind=e.kind, severity=float(e.severity), i=np.arange(len(t)),
                                 onset=t.onset.to_numpy(), y=t.label.to_numpy(),
                                 n_samples=t.n_samples.to_numpy(), valid_online=t.valid.to_numpy()))
        if with_intervals:
            ov, ins, niv = interval_hits(d, t.onset.to_numpy())
            base["iv_overlap"], base["iv_inside"], base["n_intervals"] = ov, ins, niv
        for tag in ("csp", "net"):
            f, r = cur[tag], cache[key][tag]
            assert np.array_equal(f.label.to_numpy(), t.label.to_numpy())
            base[f"pred_{tag}"] = pd.to_numeric(f.pred, errors="coerce").to_numpy()
            base[f"valid_{tag}"] = f.valid.to_numpy()
            base[f"ref_pred_{tag}"] = pd.to_numeric(r.pred, errors="coerce").to_numpy()
            base[f"ref_valid_{tag}"] = r.valid.to_numpy()
        rows.append(base)
    D = pd.concat(rows, ignore_index=True)
    D["touched"] = (D.n_samples < 1000) | (D.valid_online == 0)
    for tag in ("csp", "net"):
        v, rv = D[f"valid_{tag}"] == 1, D[f"ref_valid_{tag}"] == 1
        D[f"ok_{tag}"] = (v & (D[f"pred_{tag}"] == D.y)).astype(int)
        D[f"ref_ok_{tag}"] = (rv & (D[f"ref_pred_{tag}"] == D.y)).astype(int)
        D[f"chg_{tag}"] = (~(v & rv & (D[f"pred_{tag}"] == D[f"ref_pred_{tag}"]))).astype(int)
    return D
