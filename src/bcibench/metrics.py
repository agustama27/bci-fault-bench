"""Lectura de una campaña y cálculo de las variables dependientes (Métodos).

Familias: integridad, temporalidad, disponibilidad, desempeño funcional,
observabilidad. Unidad estadística = sujeto (mediana sobre corridas).

Definiciones operativas:
- balanced accuracy móvil: sobre una ventana de W ensayos válidos consecutivos.
- referencia del sujeto: distribución de la móvil sobre sus corridas sin fallo.
- umbral: percentil `q` de esa referencia (dispersión de la referencia).
- estado operativo (por segundo): llegaron muestras y no hubo excepciones.
- degradación funcional (por segundo): la última móvil disponible < umbral.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

KINDS = ["loss", "jitter", "delay", "disconnect", "burst_trial", "disconnect_trial"]
KIND_ES = {"none": "referencia", "loss": "pérdida de muestras", "jitter": "jitter",
           "delay": "retraso", "disconnect": "desconexión",
           "burst_trial": "pérdida contigua en el ensayo", "disconnect_trial": "desconexión en el ensayo"}


# ---------------------------------------------------------------- carga
def load_campaign(root: str, trials_file: str = "trials.csv") -> tuple[pd.DataFrame, dict, dict]:
    """Devuelve (ejecuciones, trials por exec_id, telemetría por exec_id).

    trials_file: "trials.csv" (decisiones en línea, CSP+LDA) o "trials_eegnet.csv" /
    "trials_csp_offline.csv" (re-decodificación fuera de línea de los segmentos guardados).
    """
    execs, trials, tele = [], {}, {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        pj, cj = os.path.join(d, "producer.json"), os.path.join(d, "consumer.json")
        if not (os.path.exists(pj) and os.path.exists(cj)):
            continue
        p, c = json.load(open(pj)), json.load(open(cj))
        eid = p["exec_id"]
        tp = os.path.join(d, trials_file)
        if not os.path.exists(tp):
            continue
        tr = pd.read_csv(tp)
        if "decision_delay" not in tr.columns:
            tr["decision_delay"] = np.nan
        te = pd.read_csv(os.path.join(d, "telemetry.csv"))
        # la telemetría termina en la duración real de la corrida (ejecuciones
        # anteriores a la corrección del consumidor escribían hasta el tope)
        te = te[te["sec"] < int(np.ceil(p["duration_s"]))].reset_index(drop=True)
        trials[eid], tele[eid] = tr, te
        f = p["fault"]
        # familia trace: cada traza es su propio "tipo de fallo" (kind = trace:<nombre>);
        # la severidad es el factor de escala (1 = tal como se midió)
        kind = f"trace:{p.get('trace', f['mode'].split('@')[0])}" if f["kind"] == "trace" else f["kind"]
        if kind.startswith("trace:") and kind not in KINDS:
            KINDS.append(kind); KIND_ES[kind] = "traza " + kind[6:]
        execs.append(dict(
            exec_id=eid, subject=p["subject"], run=str(p["run"]), kind=kind,
            severity=f["severity"], mode=f["mode"], label=p["label"],
            n_pushed=p["n_pushed"], n_dropped=p["n_dropped"], outages=len(p["outages"]),
            timing_p99_ms=p["timing_err_ms"]["p99"],
            n_trials=c["n_trials"], n_valid=c["n_valid"], exceptions=c["exceptions"],
            **integrity_metrics(te, p),
            **functional_metrics(tr),
        ))
    return pd.DataFrame(execs), trials, tele


def integrity_metrics(te: pd.DataFrame, p: dict) -> dict:
    """Integridad, temporalidad y disponibilidad agregadas por ejecución."""
    exp = te["n_expected"].sum()
    got = te["n_samples"].sum()
    lat = te["lat_mean"].dropna()
    return dict(
        # integridad
        recv_ratio=got / exp if exp else np.nan,
        n_gaps=int(te["n_gaps"].sum()), gap_s=float(te["gap_s"].sum()),
        # temporalidad (ms)
        lat_mean_ms=float(lat.mean() * 1e3) if len(lat) else np.nan,
        lat_max_ms=float(te["lat_max"].max() * 1e3),
        ia_std_ms=float(te["ia_std"].dropna().mean() * 1e3),
        ia_max_ms=float(te["ia_max"].max() * 1e3),
        # disponibilidad
        secs_no_data=int((te["n_samples"] == 0).sum()),
        secs_total=int(len(te)),
        reconnections=int(p.get("reconnections", 0)),
    )


def functional_metrics(tr: pd.DataFrame) -> dict:
    v = tr[tr["valid"] == 1]
    if len(v) == 0:
        return dict(bacc=np.nan, acc=np.nan, conf=np.nan, valid_rate=0.0, decision_delay_ms=np.nan)
    from sklearn.metrics import balanced_accuracy_score
    return dict(
        bacc=float(balanced_accuracy_score(v["label"], v["pred"].astype(int))),
        acc=float((v["label"] == v["pred"].astype(int)).mean()),
        conf=float(v["proba"].mean()),
        valid_rate=float(len(v) / len(tr)),
        decision_delay_ms=float(v["decision_delay"].median() * 1e3),
    )


# ---------------------------------------------------------------- ventana móvil
def rolling_bacc(tr: pd.DataFrame, W: int) -> pd.DataFrame:
    """Por ensayo: balanced accuracy de los últimos W ensayos (inválido = error).

    Devuelve columnas: onset (s desde t0), t_end (onset+6), rbacc.
    Un ensayo inválido (sin decisión) cuenta como error: el pipeline no entregó.
    """
    lab = tr["label"].to_numpy()
    pred = np.where(tr["valid"] == 1, pd.to_numeric(tr["pred"], errors="coerce").fillna(-1), -1).astype(int)
    out = []
    for i in range(len(tr)):
        if i + 1 < W:
            out.append(np.nan); continue
        l, p = lab[i + 1 - W:i + 1], pred[i + 1 - W:i + 1]
        recalls = []
        for c in np.unique(l):
            m = l == c
            recalls.append((p[m] == c).mean())
        out.append(float(np.mean(recalls)))
    return pd.DataFrame(dict(onset=tr["onset"], t_end=tr["onset"] + 6.0, rbacc=out))


def reference_threshold(ref_rolls: list[pd.DataFrame], q: float, rule: str = "percentile") -> float:
    """Umbral de degradación a partir de la dispersión de la referencia del sujeto.

    rule = "percentile": percentil q de la móvil de referencia (marca ~q % de la
           referencia como degradada por construcción).
    rule = "min": por debajo del mínimo observado sin fallo (nada de la referencia
           queda marcado; degradación = peor que todo lo visto sin fallo).
    rule = "mad": mediana − q · MAD (q actúa como factor, p. ej. 2 o 3).
    """
    vals = np.concatenate([r["rbacc"].dropna().to_numpy() for r in ref_rolls])
    if not len(vals):
        return np.nan
    if rule == "min":
        return float(vals.min())
    if rule == "mad":
        med = np.median(vals)
        mad = np.median(np.abs(vals - med)) * 1.4826
        return float(med - q * mad)
    return float(np.percentile(vals, q))


# ---------------------------------------------------------------- ventanas por segundo (RQ2, RQ3)
def window_table(te: pd.DataFrame, roll: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Una fila por segundo con telemetría, estado operativo y degradación."""
    w = te.copy()
    w["operational"] = ((w["n_samples"] > 0) & (w["exceptions"] == 0)).astype(int)
    # última móvil disponible al final de cada segundo
    r = roll.dropna().sort_values("t_end")
    idx = np.searchsorted(r["t_end"].to_numpy(), w["sec"].to_numpy() + 1.0, side="right") - 1
    last = np.where(idx >= 0, r["rbacc"].to_numpy()[np.clip(idx, 0, max(len(r) - 1, 0))] if len(r) else np.nan, np.nan)
    w["rbacc"] = last
    w["degraded"] = np.where(np.isnan(last), np.nan, (last < thr).astype(float))
    w["recv_ratio"] = w["n_samples"] / w["n_expected"]
    return w


def divergence(w: pd.DataFrame) -> dict:
    """Proporción de ventanas en que el estado operativo y la degradación divergen."""
    v = w.dropna(subset=["degraded"])
    if len(v) == 0:
        return dict(n_windows=0, silent=np.nan, loud=np.nan, divergent=np.nan)
    silent = ((v["operational"] == 1) & (v["degraded"] == 1)).mean()   # parece sano, decodifica mal
    loud = ((v["operational"] == 0) & (v["degraded"] == 0)).mean()     # parece roto, decodifica bien
    return dict(n_windows=int(len(v)), silent=float(silent), loud=float(loud), divergent=float(silent + loud))


FEATURES = ["recv_ratio", "n_gaps", "gap_s", "lat_mean", "lat_max", "ia_std", "ia_max", "exceptions"]


def subject_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mediana por (sujeto, label) sobre corridas: la observación que entra a las pruebas."""
    return (df.groupby(["subject", "kind", "severity", "label"])[cols]
              .median().reset_index())
